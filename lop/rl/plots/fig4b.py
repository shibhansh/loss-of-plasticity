import yaml
import scipy
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def bootstrapped_val(x, stride, total_steps, confidence_level=0.95, to_bootstrap=True):
    num_runs = len(x)
    avg_ret = np.zeros(total_steps // stride)
    steps = np.arange(stride, total_steps + stride, stride)
    min_rets, max_rets = np.zeros(total_steps // stride), np.zeros(total_steps // stride)
    boot_strapped_ret_low, boot_strapped_ret_high = np.zeros(total_steps // stride), np.zeros(total_steps // stride)
    for i in tqdm(range(0, total_steps // stride)):
        rets = []
        for run in range(num_runs):
            rets.append(np.abs(x[run][i*stride:(i+1)*stride]).mean())
        rets = np.array([rets])
        avg_ret[i] = rets.mean()
        min_rets[i], max_rets[i] = rets.min(), rets.max()
        if to_bootstrap and num_runs>1:
            bos = scipy.stats.bootstrap(data=(rets[0, :],), statistic=np.mean, confidence_level=confidence_level)
            boot_strapped_ret_low[i], boot_strapped_ret_high[i] = bos.confidence_interval.low, bos.confidence_interval.high
    return steps, avg_ret, min_rets, max_rets, boot_strapped_ret_low, boot_strapped_ret_high


def get_param_performance(runs, data_dir='', to_plot='pol_features_activity'):
    per_param_setting_performance, per_param_setting_termination, per_param_setting_val = [], [], []
    for idx in runs:
        file = data_dir + str(idx)
        if file[0] == 'd':  file = '../'+file
        try:
            with open(file, 'rb+') as f:
                print(f)
                data = pickle.load(f)
        except:
            with open(file+'.log', 'rb+') as f:
                print(f)
                data = pickle.load(f)
        if to_plot == 'action_output':
            legal_actions = np.logical_and(data['action_output'] < 1, -1 < data['action_output'])
            per_param_setting_val.append(legal_actions)
        elif to_plot == 'pol_features':
            print(data['pol_features'])
            per_param_setting_val.append(np.array(data['pol_features'][:, 1]))
        elif to_plot in ['pol_weights', 'val_weights']:
            print(data[to_plot])
            per_param_setting_val.append(np.array(data[to_plot][:50000, 1]))
        elif to_plot == 'pol_features_activity':
            threshold=0.01
            print(data['pol_features_activity'][1:50000, :, :]<=threshold)
            per_param_setting_val.append(np.array((data['pol_features_activity'][1:50000, :, :]<=threshold).float().mean(axis=(1, 2))))
        elif to_plot == 'stable_rank':
            print(data['stable_rank'])
            per_param_setting_val.append(np.array(data['stable_rank'][1:5000]/2.56))

    return per_param_setting_val


def plot_for_one_cfg(cfg, runs, m, ts, color='C0', min_max=False, to_plot='pol_features_activity'):
    data_dir = cfg['dir']
    val = get_param_performance(data_dir=data_dir, runs=runs, to_plot=to_plot)
    x, y, min_y, max_y, boot_strapped_ret_low, boot_strapped_ret_high = \
        bootstrapped_val(x=val, stride=m, total_steps=ts)
    plt.plot(x, y, '-', linewidth=1, color=color, label=cfg['label'])
    plt.fill_between(x, boot_strapped_ret_low, boot_strapped_ret_high, alpha=0.3, color=color)
    if min_max:
        plt.fill_between(x, min_y, max_y, alpha=0.1, color=color)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attribute', required=False, type=str, default='pol_features_activity')
    parser.add_argument('--env', required=False, type=str, default='ant')

    args = parser.parse_args()
    env = args.env
    to_plot = args.attribute

    cfg_file = f'../cfg/{env}/std.yml'
    cfg_file1 = f'../cfg/{env}/cbp.yml'
    cfg_file2 = f'../cfg/{env}/ns.yml'
    cfg_file3 = f'../cfg/{env}/l2.yml'

    cfg_files = [cfg_file, cfg_file1, cfg_file2, cfg_file3]
    colors = ['C3', 'C0', 'C1', 'C4']
    cfgs = []
    for file in cfg_files:
        if file == '':  continue
        cfgs.append(yaml.safe_load(open(file)))

    num_runs = 30
    runs = [i + 0 for i in range(0, num_runs)]
    m = 100 * 1000
    ts = 50 * 1000 * 1000
    fig, ax = plt.subplots()

    if to_plot == 'weight_change':
        ts, m, max_slicing = 95, 1, 500
    if to_plot in ['pol_weights', 'pol_features_activity', 'val_weights']:
        ts, m = ts//1000, m//1000
    if to_plot == 'stable_rank':
        ts, m = ts//10000, m//10000

    for idx, cfg in enumerate(cfgs):
        plot_for_one_cfg(cfg=cfg, runs=runs, m=m, ts=ts, color=colors[idx], to_plot=to_plot)

    xticks = [0, 0.5 * ts, ts]

    if to_plot == 'return': yticks = [0, 500, 1000, 1500, 2000, 5000]
    elif to_plot in ['pol_features', 'action_output', 'pol_features_activity']:  yticks = [0, 0.2, 0.4, 0.6]
    elif to_plot == 'weight_change':
        yticks = [5, 10, 15, 20]
    elif to_plot == 'val_weights':
        yticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    elif to_plot == 'pol_weights':
        yticks = [0, 0.05, 0.1, 0.15]
    elif to_plot == 'stable_rank':
        yticks = [25, 50, 75, 100]

    fontsize = 15
    ax.set_xticks(xticks)
    ax.set_xticklabels(['' for _ in xticks], fontsize=fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks(yticks)
    ax.set_yticklabels(['' for _ in yticks], fontsize=fontsize)
    ax.set_ylim(yticks[0], yticks[-1])

    ax.yaxis.grid()

    plt.savefig('fig4b.png', bbox_inches='tight', dpi=250)
    plt.close()


if __name__ == "__main__":
    main()

