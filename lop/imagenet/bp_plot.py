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
        file = params['data_dir'] + str(setting_idx) + '/' + str(idx)
        with open(file, 'rb') as f:
            data = pickle.load(f)

        # Online performance
        per_param_setting_performance.append(np.array(bin_m_errs(errs=data['test_accuracies'][:, -1].flatten()*100, m=m)))
    print(param_settings[setting_idx], setting_idx, np.array(per_param_setting_performance).mean())
    return np.array(per_param_setting_performance)


def main(arguments):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg_file', help="Path of the file containing the parameters of the experiment", type=str,
                            default='cfg/bp.json')
    args = parser.parse_args(arguments)
    cfg_file = args.cfg_file

    with open(cfg_file, 'r') as f:
        params = json.load(f)
    list_params, param_settings = get_configurations(params=params)

    performances = []
    m = 50
    num_runs = params['num_runs']
    num_settings = len(param_settings)

    indices = [i for i in range(num_settings)]
    for i in indices:
        performances.append(add_cfg_performance(cfg=cfg_file, setting_idx=i, m=m, num_runs=num_runs))

    performances.append(0.771 * np.ones(performances[-1].shape) * 100)
    param_settings.append('linear')
    indices.append(-1)

    yticks = [70, 75, 80, 85, 90]
    generate_online_performance_plot(
        performances=performances,
        colors=['C3', 'C1', 'C2'],
        yticks=yticks,
        xticks=[0, 1000, 2000],
        xticks_labels=['0', '1k', '2k'],
        m=m,
        fontsize=18,
        labels=np.array(param_settings)[indices],
    )


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

