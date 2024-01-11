import torch
from torch import nn
from torch.distributions import Normal

from lop.utils.miscellaneous import net_init, fc_body, register_hook, kaiming_init, xavier_init, lecun_init, gaussian_init


class Policy(object):
    def action(self, x, to_log_features=False):
        """
        :param x: tensor of shape [N, 1], where N is number of observations
        :return:
            action: of shape [N, 1]
            lprob: of shape [N, 1]
        """
        with torch.no_grad():
            dist = self.dist(x, to_log_features=to_log_features)
            action = dist.sample()
            lprob = dist.log_prob(action).sum(-1, keepdim=True)
        return action, lprob, dist

    def dist(self, x, to_log_features):
        pass

    def dist_to(self, dist, to_device='cpu'):
        pass

    def dist_stack(self, dists):
        pass

    def dist_index(self, dist, ind):
        pass


class MLPPolicy(Policy, nn.Module):
    def __init__(self, o_dim, a_dim, act_type='Tanh', h_dim=(50,), log_std=0, device='cpu', init='kaiming', bias=True,
                 std_dev=1e-1, output_tanh=False):
        super().__init__()
        self.act_type = act_type
        self.device = device
        mean_net = fc_body(act_type, o_dim, h_dim, bias=bias)
        if len(h_dim) > 0:
            mean_net.append(nn.Linear(h_dim[-1], a_dim, bias=bias))
        else:
            mean_net.append(nn.Linear(o_dim, a_dim, bias=bias))
        if output_tanh:
            mean_net.append(nn.Tanh())
        self.mean_net = nn.Sequential(*mean_net)
        if init == 'kaiming':
            kaiming_init(self.mean_net, act=act_type, bias=bias)
        elif init == 'xavier':
            xavier_init(self.mean_net, act=act_type, bias=bias)
        elif init == 'lecun':
            lecun_init(self.mean_net, bias=bias)
        elif init == 'default':
            net_init(self.mean_net)
        elif init == 'gaussian':
            gaussian_init(self.mean_net, std_dev=std_dev)
        self.log_std = nn.Parameter(torch.ones(a_dim) * log_std)
        self.to(device)
        self.discrete_actions = False
        # Setup feature logging
        self.setup_feature_logging(h_dim=h_dim)

    def setup_feature_logging(self, h_dim) -> None:
        """
        Input h_dim: A list describing the network architecture. Ex. [64, 64] describes a network with two hidden layers
                        of size 64 each.
        """
        self.to_log_features = False
        self.activations = {}
        self.feature_keys = [self.mean_net[i * 2 + 1] for i in range(len(h_dim))]

        def hook_fn(m, i, o):
            if self.to_log_features:
                self.activations[m] = o

        # Register "hook_fn" for each layer
        register_hook(self.mean_net, hook_fn)

    def get_activations(self):
        return [self.activations[key] for key in self.feature_keys]

    def logp_dist(self, x, a, to_log_features=False):
        dist = self.dist(x, to_log_features=to_log_features)
        lprob = dist.log_prob(torch.as_tensor(a, device=self.device)).sum(1, keepdim=True)
        return lprob, dist

    def dist(self, x, to_log_features=False):
        x = x.to(self.device)
        self.to_log_features = to_log_features
        action_mean = self.mean_net(x)
        self.to_log_features = False
        return Normal(action_mean, torch.exp(self.log_std))

    def dist_to(self, dist, to_device='cpu'):
        dist.loc.to(to_device)
        dist.scale.to(to_device)
        return dist

    def dist_stack(self, dists, device='cpu'):
        return Normal(
            torch.cat(tuple([dists[i].loc for i in range(len(dists))])).to(device),
            torch.cat(tuple([dists[i].scale for i in range(len(dists))])).to(device)
        )

    def dist_index(self, dist, ind):
        return Normal(dist.loc[ind], dist.scale[ind])