import os
import yaml
import pickle
import argparse
import subprocess
import numpy as np

import gym
import torch
from torch.optim import Adam

import lop.envs
from lop.algos.rl.buffer import Buffer
from lop.nets.policies import MLPPolicy
from lop.nets.valuefs import MLPVF
from lop.algos.rl.agent import Agent
from lop.algos.rl.ppo import PPO
from lop.utils.miscellaneous import compute_matrix_rank_summaries


def save_data(cfg, rets, termination_steps,
              pol_features_activity, stable_rank, mu, pol_weights, val_weights,
              action_probs=None, weight_change=[], friction=-1.0, num_updates=0, previous_change_time=0):
    data_dict = {
        'rets': np.array(rets),
        'termination_steps': np.array(termination_steps),
        'pol_features_activity': pol_features_activity,
        'stable_rank': stable_rank,
        'action_output': mu,
        'pol_weights': pol_weights,
        'val_weights': val_weights,
        'action_probs': action_probs,
        'weight_change': torch.tensor(weight_change).numpy(),
        'friction': friction,
        'num_updates': num_updates,
        'previous_change_time': previous_change_time
    }
    with open(cfg['log_path'], 'wb') as f:
        pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)


def load_data(cfg):
    with open(cfg['log_path'], 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict


def save_checkpoint(cfg, step, learner):
    # Save step, model and optimizer states
    ckpt_dict = dict(
        step = step,
        actor = learner.pol.state_dict(),
        critic = learner.vf.state_dict(),
        opt = learner.opt.state_dict()
    )
    torch.save(ckpt_dict, cfg['ckpt_path'])
    print(f'Save checkpoint at step={step}')


def load_checkpoint(cfg, device, learner):
    # Load step, model and optimizer states
    step = 0
    ckpt_dict = torch.load(cfg['ckpt_path'], map_location=device)
    step = ckpt_dict['step']
    learner.pol.load_state_dict(ckpt_dict['actor'])
    learner.vf.load_state_dict(ckpt_dict['critic'])
    learner.opt.load_state_dict(ckpt_dict['opt'])
    print(f"Successfully restore from checkpoint: {cfg['ckpt_path']}.")
    return step, learner


def main():
    # Setup
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=False, type=str, default='./cfg/ant/std.yml')
    parser.add_argument('-s', '--seed', required=False, type=int, default="1")
    parser.add_argument('-d', '--device', required=False, default='cpu')

    args = parser.parse_args()
    if args.device: device = args.device
    else: device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = yaml.safe_load(open(args.config))
    cfg['seed'] = args.seed
    cfg['log_path'] = cfg['dir'] + str(args.seed) + '.log'
    cfg['ckpt_path'] = cfg['dir'] + str(args.seed) + '.pth'
    cfg['done_path'] = cfg['dir'] + str(args.seed) + '.done'

    bash_command = "mkdir -p " + cfg['dir']
    subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)

    # Set default values
    cfg.setdefault('wd', 0)
    cfg.setdefault('init', 'lecun')
    cfg.setdefault('to_log', [])
    cfg.setdefault('beta_1', 0.9)
    cfg.setdefault('beta_2', 0.999)
    cfg.setdefault('eps', 1e-8)
    cfg.setdefault('no_clipping', False)
    cfg.setdefault('loss_type', 'ppo')
    cfg.setdefault('frictions_file', 'cfg/frictions')
    cfg.setdefault('max_grad_norm', 1e9)
    cfg.setdefault('perturb_scale', 0)
    cfg['n_steps'] = int(float(cfg['n_steps']))
    cfg['perturb_scale'] = float(cfg['perturb_scale'])
    n_steps = cfg['n_steps']    

    # Set default values for CBP
    cfg.setdefault('mt', 10000)
    cfg.setdefault('rr', 0)
    cfg['rr'] = float(cfg['rr'])
    cfg.setdefault('decay_rate', 0.99)
    cfg.setdefault('redo', False)
    cfg.setdefault('threshold', 0.03)
    cfg.setdefault('reset_period', 1000)
    cfg.setdefault('util_type_val', 'contribution')
    cfg.setdefault('util_type_pol', 'contribution')
    cfg.setdefault('pgnt', (cfg['rr']>0) or cfg['redo'])
    cfg.setdefault('vgnt', (cfg['rr']>0) or cfg['redo'])

    # Initialize env
    seed = cfg['seed']
    friction = -1.0
    if cfg['env_name'] in ['SlipperyAnt-v2', 'SlipperyAnt-v3']:
        xml_file = os.path.abspath(cfg['dir'] + f'slippery_ant_{seed}.xml')
        cfg.setdefault('friction', [0.02, 2])
        cfg.setdefault('change_time', int(2e6))

        with open(cfg['frictions_file'], 'rb+') as f:
            frictions = pickle.load(f)
        friction_number = 0
        new_friction = frictions[seed][friction_number]

        if friction < 0: # If no saved friction, use the default value 1.0
            friction = 1.0
        env = gym.make(cfg['env_name'], friction=new_friction, xml_file=xml_file)
        print(f'Initial friction: {friction:.6f}')
    else:
        env = gym.make(cfg['env_name'])
    env.name = None

    # Set random seeds
    np.random.seed(seed)
    random_state = np.random.get_state()
    torch_seed = np.random.randint(1, 2 ** 31 - 1)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    
    # Initialize algorithm
    opt = Adam
    num_layers = len(cfg['h_dim'])
    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    pol = MLPPolicy(o_dim, a_dim, act_type=cfg['act_type'], h_dim=cfg['h_dim'], device=device, init=cfg['init'])
    vf = MLPVF(o_dim, act_type=cfg['act_type'], h_dim=cfg['h_dim'], device=device, init=cfg['init'])
    np.random.set_state(random_state)
    buf = Buffer(o_dim, a_dim, cfg['bs'], device=device)

    learner = PPO(pol, buf, cfg['lr'], g=cfg['g'], vf=vf, lm=cfg['lm'], Opt=opt,
                  u_epi_up=cfg['u_epi_ups'], device=device, n_itrs=cfg['n_itrs'], n_slices=cfg['n_slices'],
                  u_adv_scl=cfg['u_adv_scl'], clip_eps=cfg['clip_eps'],
                  max_grad_norm=cfg['max_grad_norm'], init=cfg['init'],
                  wd=float(cfg['wd']),
                  betas=(cfg['beta_1'], cfg['beta_2']), eps=float(cfg['eps']), no_clipping=cfg['no_clipping'],
                  loss_type=cfg['loss_type'], perturb_scale=cfg['perturb_scale'],
                  util_type_val=cfg['util_type_val'], replacement_rate=cfg['rr'], decay_rate=cfg['decay_rate'],
                  vgnt=cfg['vgnt'], pgnt=cfg['pgnt'], util_type_pol=cfg['util_type_pol'], mt=cfg['mt'],
                  redo=cfg['redo'], threshold=cfg['threshold'], reset_period=cfg['reset_period']
                  )

    to_log = cfg['to_log']
    agent = Agent(pol, learner, device=device, to_log_features=(len(to_log) > 0))

    # Load checkpoint
    if os.path.exists(cfg['ckpt_path']):
        start_step, agent.learner = load_checkpoint(cfg, device, agent.learner)
    else:
        start_step = 0
    
    # Initialize log
    if os.path.exists(cfg['log_path']):
        data_dict = load_data(cfg)
        num_updates = data_dict['num_updates']
        previous_change_time = data_dict['previous_change_time']
        for k, v in data_dict.items():
            try:
                data_dict[k] = list(v)
            except:
                pass
        rets = data_dict['rets']
        termination_steps = data_dict['termination_steps']
        pol_features_activity = data_dict['pol_features_activity']
        stable_rank = data_dict['stable_rank']
        if 'pol_features_activity' in to_log:
            short_term_feature_activity = torch.zeros(size=(1000, num_layers, cfg['h_dim'][0]))
            pol_features_activity = torch.stack(pol_features_activity)
        if 'stable_rank' in to_log:
            stable_rank = torch.stack(stable_rank)
        mu = data_dict['action_output']
        if 'mu' in to_log:
            mu = np.array(mu)
        pol_weights = data_dict['pol_weights']
        if 'pol_weights' in to_log:
            pol_weights = np.array(pol_weights)
        val_weights = data_dict['val_weights']
        if 'val_weights' in to_log:
            val_weights = np.array(val_weights)
        weight_change = data_dict['weight_change']
    else:
        num_updates = 0
        previous_change_time = 0
        rets, termination_steps = [], []
        mu, weight_change, pol_features_activity, stable_rank, pol_weights, val_weights = [], [], [], [], [], []
        if 'mu' in to_log:
            mu = np.ones(size=(n_steps, a_dim))
        if 'pol_weights' in to_log:
            pol_weights = np.zeros(shape=(n_steps//1000 + 2, (len(pol.mean_net)+1)//2))
        if 'val_weights' in to_log:
            val_weights = np.zeros(shape=(n_steps//1000 + 2, (len(pol.mean_net)+1)//2))
        if 'pol_features_activity' in to_log:
            short_term_feature_activity = torch.zeros(size=(1000, num_layers, cfg['h_dim'][0]))
            pol_features_activity = torch.zeros(size=(n_steps//1000 + 2, num_layers, cfg['h_dim'][0]))
        if 'stable_rank' in to_log:
            stable_rank = torch.zeros(size=(n_steps//10000 + 2,))

    ret = 0
    epi_steps = 0
    o = env.reset()
    print('start_step:', start_step)
    # Interaction loop
    for step in range(start_step, n_steps):
        a, logp, dist, new_features = agent.get_action(o)
        op, r, done, infos = env.step(a)
        epi_steps += 1
        op_ = op
        val_logs = agent.log_update(o, a, r, op_, logp, dist, done)
        # Logging
        with torch.no_grad():
            if 'weight_change' in to_log and 'weight_change' in val_logs.keys(): weight_change.append(val_logs['weight_change'])
            if 'mu' in to_log: mu[step] = a
            if step % 1000 == 0:
                if step % 10000 == 0 and 'stable_rank' in to_log:
                    _, _, _, stable_rank[step//10000] = compute_matrix_rank_summaries(m=short_term_feature_activity[:, -1, :], use_scipy=True)
                if 'pol_features_activity' in to_log:
                    pol_features_activity[step//1000] = (short_term_feature_activity>0).float().mean(dim=0)
                    short_term_feature_activity *= 0
                if 'pol_weights' in to_log:
                    for layer_idx in range((len(pol.mean_net) + 1) // 2):
                        pol_weights[step//1000, layer_idx] = pol.mean_net[2 * layer_idx].weight.data.abs().mean()
                if 'val_weights' in to_log:
                    for layer_idx in range((len(learner.vf.v_net) + 1) // 2):
                        val_weights[step//1000, layer_idx] = learner.vf.v_net[2 * layer_idx].weight.data.abs().mean()
            if 'pol_features_activity' in to_log:
                for i in range(num_layers):
                    short_term_feature_activity[step % 1000, i] = new_features[i]

        o = op
        ret += r
        if done:
            # print(step, "(", epi_steps, ") {0:.2f}".format(ret))
            rets.append(ret)
            termination_steps.append(step)
            ret = 0
            epi_steps = 0
            if cfg['env_name'] in ['SlipperyAnt-v2', 'SlipperyAnt-v3'] and step - previous_change_time > cfg['change_time']:
                previous_change_time = step
                env.close()
                friction_number += 1
                new_friction = frictions[seed][friction_number]
                print(f'{step}: change friction to {new_friction:.6f}')
                env.close()
                env = gym.make(cfg['env_name'], friction=new_friction, xml_file=xml_file)
                env.name = None
                agent.env = env
            o = env.reset()

        if step % (n_steps//100) == 0 or step == n_steps-1:
            # Save checkpoint
            save_checkpoint(cfg, step, agent.learner)
            # Save data logs
            save_data(cfg=cfg, rets=rets, termination_steps=termination_steps,
                      pol_features_activity=pol_features_activity, stable_rank=stable_rank, mu=mu, pol_weights=pol_weights,
                      val_weights=val_weights, weight_change=weight_change, friction=friction,
                      num_updates=num_updates, previous_change_time=previous_change_time)

    with open(cfg['done_path'], 'w') as f:
        f.write('All done!')
        print('The experiment finished successfully!')


if __name__ == "__main__":
    main()