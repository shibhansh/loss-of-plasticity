import numpy as np
import torch
from torch import nn
from lop.algos.gnt import GnT
from lop.algos.gntRedo import GnTredo
from lop.algos.rl.learner import Learner
import torch.nn.functional as F


class PPO(Learner):
    """
    Implementation of PPO
    """
    def __init__(self, pol, buf, lr, g, vf, lm,
                 Opt,
                 device='cpu',
                 u_epi_up=0,  # whether update epidocially or not
                 n_itrs=10,
                 n_slices=32,
                 u_adv_scl=1,  # scale return with mean and std
                 clip_eps=0.2,
                 max_grad_norm=int(1e9),  # maximum gradient norm for clip_grad_norm
                 util_type_val='contribution',
                 util_type_pol='contribution',
                 replacement_rate=1e-4,
                 decay_rate=0.99,
                 mt=10000,
                 vgnt=0,
                 pgnt=0,
                 init='lecun',
                 wd=0,
                 perturb_scale=0,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 no_clipping=False,
                 loss_type='ppo',
                 redo=False,
                 threshold=0.03,
                 reset_period=1000,
                 ):
        self.pol = pol
        self.buf = buf
        self.g = g
        self.vf = vf
        self.lm = lm
        self.opt = Opt(list(self.pol.parameters()) + list(self.vf.parameters()), lr=lr, weight_decay=wd, betas=betas, eps=eps)
        self.device =device
        self.u_epi_up = u_epi_up
        self.n_itrs = n_itrs
        self.n_slices = n_slices
        self.u_adv_scl = u_adv_scl
        self.clip_eps = clip_eps
        self.max_grad_norm = max_grad_norm
        self.vgnt = vgnt
        self.pgnt = pgnt
        self.perturb_scale = perturb_scale
        self.no_clipping = no_clipping
        self.loss_type = loss_type
        self.to_perturb = self.perturb_scale != 0

        """
        Initialize a generate-and-test object for your network
        """

        self.pol_gnt = GnT(net=self.pol.mean_net, hidden_activation=self.pol.act_type.lower(), opt=self.opt,
                            replacement_rate=replacement_rate, decay_rate=decay_rate, init=init, device=device,
                            maturity_threshold=mt, util_type=util_type_pol, loss_func='ppo')
        self.val_gnt = GnT(net=self.vf.v_net, hidden_activation=self.vf.act_type.lower(), opt=self.opt,
                            replacement_rate=replacement_rate, decay_rate=decay_rate, init=init,
                            maturity_threshold=mt, util_type=util_type_val, loss_func=F.mse_loss,
                            device=device)
        if redo:
            self.pol_gnt = GnTredo(net=self.pol.mean_net, hidden_activation=self.pol.act_type.lower(),
                                   threshold=threshold, device=device, init=init, reset_period=reset_period)
            self.val_gnt = GnTredo(net=self.vf.v_net, hidden_activation=self.vf.act_type.lower(),
                                   threshold=threshold, device=device, init=init, reset_period=reset_period)


    def log(self, o, a, r, op, logpb, dist, done):
        self.buf.store(o, a, r, op, logpb, dist, done)

    def learn_time(self, done):
        return (not self.u_epi_up or done) and len(self.buf.o_buf) >= self.buf.bs

    def post_learn(self):
        self.buf.clear()

    def perturb(self, net, device='cpu'):
        with torch.no_grad():
            for i in range(int(len(net) / 2)+1):
                net[i * 2].bias += torch.empty(net[i * 2].bias.shape, device=device).normal_(mean=0, std=self.perturb_scale)
                net[i * 2].weight += torch.empty(net[i * 2].weight.shape, device=device).normal_(mean=0, std=self.perturb_scale)

    def get_rets_advs(self, rs, dones, vals, device='cpu'):
        dones, rs, vals = dones.to(device), rs.to(device), vals.to(device)
        advs = torch.as_tensor(np.zeros(len(rs)+1, dtype=np.float32), device=device)
        for t in reversed(range(len(rs))):
            delta = rs[t] + (1-dones[t])*self.g*vals[t+1] - vals[t]
            advs[t] = delta + (1-dones[t])*self.g*self.lm*advs[t+1]
        v_rets = advs[:-1] + vals[:-1]
        advs = advs[:-1].view(-1, 1)
        if self.u_adv_scl:
            advs = advs - advs.mean()
            if advs.std() != 0 and not torch.isnan(advs.std()): advs /= advs.std()
        v_rets, advs = v_rets.detach().to(self.device), advs.detach().to(self.device)
        return v_rets.view(-1, 1), advs

    def learn(self):
        os, acts, rs, op, logpbs, _, dones = self.buf.get(self.pol.dist_stack)
        with torch.no_grad():
            pre_vals = self.vf.value(torch.cat((os, op)))
        v_rets, advs = self.get_rets_advs(rs, dones, pre_vals.t()[0])
        inds = np.arange(os.shape[0])
        mini_bs = self.buf.bs // self.n_slices
        iter_num = -1

        old_weights = []
        for layer in self.pol.mean_net:
            if type(layer) is torch.nn.modules.linear.Linear:
                old_weights.append(torch.clone(layer.weight.data))

        for _ in range(self.n_itrs):
            np.random.shuffle(inds)
            for start in range(0, len(os), mini_bs):
                iter_num += 1
                ind = inds[start:start + mini_bs]
                assert self.loss_type == 'ppo', 'Only PPO loss is supported'
                # Calculate policy loss
                logpts, _ = self.pol.logp_dist(os[ind], acts[ind], to_log_features=True)
                grad_sub = (logpts - logpbs[ind]).exp()
                p_loss0 = - (grad_sub * advs[ind])
                if self.no_clipping:
                    p_loss = p_loss0
                else:
                    ext_loss = - (torch.clamp(grad_sub, 1 - self.clip_eps, 1 + self.clip_eps) * advs[ind])
                    p_loss = torch.max(p_loss0, ext_loss)
                p_loss = p_loss.mean()
                # Calculate value loss
                vals = self.vf.value(os[ind], to_log_features=True)
                v_loss = (v_rets[ind] - vals).pow(2).mean()
                # Backprop
                p_loss += v_loss # no value loss weight applied
                self.opt.zero_grad()
                p_loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(list(self.pol.parameters()) + list(self.vf.parameters()), self.max_grad_norm)
                self.opt.step()
                if self.to_perturb:
                    self.perturb(net=self.pol.mean_net)
                    self.perturb(net=self.vf.v_net)
                # Selective reinitialization
                if self.pgnt:
                    with torch.no_grad():
                        if isinstance(self.pol_gnt, GnT):
                            self.pol_gnt.gen_and_test(features=self.pol.get_activations()+[None])
                        elif isinstance(self.pol_gnt, GnTredo):
                            self.pol_gnt.gen_and_test(features_history=torch.stack(self.pol.get_activations()).permute(1, 0, 2))
                if self.vgnt:
                    with torch.no_grad():
                        if isinstance(self.val_gnt, GnT):
                            self.val_gnt.gen_and_test(features=self.vf.get_activations()+[None])
                        elif isinstance(self.pol_gnt, GnTredo):
                            features_history = torch.stack(self.vf.get_activations()).permute(1, 0, 2)
                            self.val_gnt.gen_and_test(features_history=features_history)
        idx, change = 0, 0
        for layer in self.pol.mean_net:
            if type(layer) is torch.nn.modules.linear.Linear:
                change += (old_weights[idx] - layer.weight.data).abs().sum()
                idx += 1

        return {'weight_change': change}
