import yaml
import scipy
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_max(x, m=100):
    lex_x = x.shape[0]
    max_x = np.zeros(lex_x//m)
    for i in range(max_x.shape[0]):
        max_x[i] = np.max(x[i*m: (i+1)*m])
    return max_x


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


def get_param_performance(runs, data_dir='', m=1):
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
        per_param_setting_performance.append(np.array(data['rets']))
        per_param_setting_termination.append(np.array(data['termination_steps']))
        print(data['termination_steps'][-1])
        per_param_setting_val.append(get_max(data['weight_change'], m=m))

    return per_param_setting_termination, per_param_setting_performance, per_param_setting_val


def plot_for_one_cfg(cfg, runs, m, ts, color='C0', max_slicing=100):
    data_dir = cfg['dir']
    terminations, returns, val = get_param_performance(data_dir=data_dir, runs=runs, m=max_slicing)
    x, y, min_y, max_y, boot_strapped_ret_low, boot_strapped_ret_high = bootstrapped_val(x=val, stride=m, total_steps=ts)
    plt.plot(x, y, '-', linewidth=1, color=color, label=cfg['label'])
    plt.fill_between(x, boot_strapped_ret_low, boot_strapped_ret_high, alpha=0.3, color=color)


def main():
    env = 'ant'
    cfg_file = f'../cfg/{env}/std.yml'
    cfg_file1 = f'../cfg/{env}/ns.yml'

    cfg_files = [cfg_file, cfg_file1]
    colors = ['C3', 'C1']
    cfgs = []
    ts = 0
    for file in cfg_files:
        if file == '':  continue
        cfgs.append(yaml.safe_load(open(file)))
        ts = max(ts, float(cfgs[-1]['n_steps']))
        if 'label' not in cfgs[-1].keys():   cfgs[-1]['label'] = ''

    num_runs = 30
    runs = [i + 0 for i in range(0, num_runs)]
    fig, ax = plt.subplots()
    ts, m, max_slicing = 20, 1, 500

    for idx, cfg in enumerate(cfgs):
        plot_for_one_cfg(cfg=cfg, runs=runs, m=m, ts=ts, color=colors[idx], max_slicing=max_slicing)

    xticks = [0, 0.5 * ts, ts]
    yticks = [0, 50, 100, 120]

    ax.set_xticks(xticks)
    ax.set_xticklabels(['' for _ in xticks])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks(yticks)
    ax.set_yticklabels(['' for _ in yticks])
    ax.set_ylim(yticks[0], yticks[-1])

    ax.yaxis.grid()

    plt.savefig('c.png', bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == "__main__":
    main()