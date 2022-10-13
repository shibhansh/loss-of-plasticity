import sys
import json
import pickle
import argparse
from lop.utils.miscellaneous import *
from lop.utils.plot_online_performance import generate_online_performance_plot


def add_cfg_performance(cfg='', setting_idx=0, m=2*10*1000, num_runs=30):
    with open(cfg, 'r') as f:
        params = json.load(f)
    list_params, param_settings = get_configurations(params=params)
    per_param_setting_performance = []
    for idx in range(num_runs):
        file = '../' + params['data_dir'] + str(setting_idx) + '/' + str(idx)
        with open(file, 'rb') as f:
            data = pickle.load(f)

        # Online performance
        per_param_setting_performance.append(np.array(bin_m_errs(errs=100*data['accuracies'][:60*1000*150], m=m)))
    print(param_settings[setting_idx], setting_idx, np.array(per_param_setting_performance).mean())
    return np.array(per_param_setting_performance)


def main(arguments):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg_file', help="Path of the file containing the parameters of the experiment", type=str,
                            default='../cfg/sgd/bp/small_net.json')
    args = parser.parse_args(arguments)
    cfg_file = args.cfg_file

    with open(cfg_file, 'r') as f:
        params = json.load(f)
    list_params, param_settings = get_configurations(params=params)

    performances = []
    m = 60*1000
    num_runs = params['num_runs']

    indices = [i for i in range(3)]
    for i in indices:
        performances.append(add_cfg_performance(cfg=cfg_file, setting_idx=i, m=m, num_runs=num_runs))

    yticks = [88, 90, 92, 94, 96]
    generate_online_performance_plot(
        performances=performances,
        colors=['C1', 'C3', 'C5', 'C2', 'C4', 'C6'],
        yticks=yticks,
        xticks=[0, 75*m, 150*m],
        xticks_labels=['0', '75', '150'],
        m=m,
        fontsize=18,
        labels=param_settings,
    )


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

