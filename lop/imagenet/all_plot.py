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
        if data['test_accuracies'].shape[0] == 2000:
            data['test_accuracies'] = torch.cat((data['test_accuracies'], torch.zeros((3000,250))))
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

    performances = []
    m = 50
    num_runs = params['num_runs']
    num_runs = 30

    bp_cfg = 'cfg/bp.json'
    l2_cfg = 'cfg/l2.json'
    snp_cfg = 'cfg/snp.json'
    cbp_cfg = 'cfg/cbp.json'
    performances.append(add_cfg_performance(cfg=bp_cfg, setting_idx=0, m=m, num_runs=num_runs))
    performances.append(add_cfg_performance(cfg=l2_cfg, setting_idx=0, m=m, num_runs=num_runs))
    performances.append(add_cfg_performance(cfg=snp_cfg, setting_idx=0, m=m, num_runs=num_runs))
    performances.append(add_cfg_performance(cfg=cbp_cfg, setting_idx=0, m=m, num_runs=num_runs))
    

    yticks = [82, 84, 86, 88, 90, 92]
    generate_online_performance_plot(
        performances=performances,
        colors=['C3', 'C4', 'C1', 'C0', 'C5', 'C6', 'C7', 'C8', 'C9'],
        yticks=yticks,
        xticks=[0, 2500, 5000],
        xticks_labels=['0', '2500', '5000'],
        m=m,
        fontsize=18,
        labels=np.array(['bp', 'l2', 'snp', 'cbp']),
    )


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

