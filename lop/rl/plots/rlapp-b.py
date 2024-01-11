import yaml
import scipy
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def bootstrapped_return(x, y, stride, total_steps, confidence_level=0.95, to_bootstrap=True):
    assert len(x) == len(y)
    num_runs = len(x)
    avg_ret = np.zeros(total_steps // stride)
    steps = np.arange(stride, total_steps + stride, stride)
    min_rets, max_rets = np.zeros(total_steps // stride), np.zeros(total_steps // stride)
    boot_strapped_ret_low, boot_strapped_ret_high = np.zeros(total_steps // stride), np.zeros(total_steps // stride)
    for i in tqdm(range(0, total_steps // stride)):
        rets = []
        for run in range(num_runs):
            xa = x[run][:np.searchsorted(x[run], total_steps)+1]
            ya = y[run][:xa.shape[0]]
            rets.append(ya[np.logical_and(i*stride < xa, xa <= (i+1)*stride)].mean())
        rets = np.array([rets])
        avg_ret[i] = rets.mean()
        min_rets[i], max_rets[i] = rets.min(), rets.max()
        if to_bootstrap:
            bos = scipy.stats.bootstrap(data=(rets[0, :],), statistic=np.mean, confidence_level=confidence_level)
            boot_strapped_ret_low[i], boot_strapped_ret_high[i] = bos.confidence_interval.low, bos.confidence_interval.high
    return steps, avg_ret, min_rets, max_rets, boot_strapped_ret_low, boot_strapped_ret_high


def get_param_performance(runs, data_dir=''):
    per_param_setting_performance, per_param_setting_termination = [], []
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
        per_param_setting_performance.append(np.array(data['rets']))
        per_param_setting_termination.append(np.array(data['termination_steps']))
        print(data['termination_steps'][-1])

    return per_param_setting_termination, per_param_setting_performance


def plot_for_one_cfg(cfg, runs, m, ts, color='C0', min_max=False):
    data_dir = cfg['dir']
    terminations, returns = get_param_performance(data_dir=data_dir, runs=runs)
    x, y, min_y, max_y, boot_strapped_ret_low, boot_strapped_ret_high = \
        bootstrapped_return(x=terminations, y=returns, stride=m, total_steps=ts)
    plt.plot(x, y, '-', linewidth=1, color=color, label=cfg['label'])
    plt.fill_between(x, boot_strapped_ret_low, boot_strapped_ret_high, alpha=0.3, color=color)
    if min_max:
        plt.fill_between(x, min_y, max_y, alpha=0.1, color=color)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', required=False, type=str, default='ant')
    parser.add_argument('--all', required=False, type=bool, default=False)

    args = parser.parse_args()
    env = args.env

    cfg_file = f'../cfg/{env}/redo.yml'
    cfg_file1 = f'../cfg/{env}/cbp.yml'
    cfg_file2 = f'../cfg/{env}/l2.yml'
    cfg_files = [cfg_file, cfg_file1, cfg_file2]
    colors = ['C3', 'C0', 'C4', 'C1']
    cfgs = []
    for file in cfg_files:
        if file == '':  continue
        cfgs.append(yaml.safe_load(open(file)))
        if 'label' not in cfgs[-1].keys():   cfgs[-1]['label'] = ''

    num_runs = 30
    runs = [i + 0 for i in range(0, num_runs)]
    m = 250 * 1000
    ts = 50 * 1000 * 1000
    fig, ax = plt.subplots()

    yticks = [0, 2000, 4000, 5500]

    for idx, cfg in enumerate(cfgs):
        plot_for_one_cfg(cfg=cfg, runs=runs, m=m, ts=ts, color=colors[idx])

    xticks = [0, 0.5 * ts, ts]


    fontsize = 15
    ax.set_xticks(xticks)
    ax.set_xticklabels(['' for _ in xticks], fontsize=fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks(yticks)
    ax.set_yticklabels(['' for _ in yticks], fontsize=fontsize)
    ax.set_ylim(yticks[0], yticks[-1])

    ax.yaxis.grid()

    plt.savefig('b.png', bbox_inches='tight', dpi=250)
    plt.close()


if __name__ == "__main__":
    main()