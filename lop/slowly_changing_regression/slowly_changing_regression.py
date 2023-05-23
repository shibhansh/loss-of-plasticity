import sys
import json
import torch
import pickle
import argparse
from tqdm import tqdm
from lop.nets.fix_ltu_net import FixLTUNet


def generate_problem_data(
        flip_after=10000,
        data_file='data/env_data/0',
        num_data_points=1000*100,
        num_inputs=20,
        num_target_features=20,
        num_flipping_bits=None,
        beta=0.75,
        flip_one=False,
):
    """
    Generates data for one run on the slowly changing regression problem
    """
    target_network = FixLTUNet(
        num_inputs=num_inputs,
        num_features=num_target_features,
        beta=beta,
    )

    num_flips = int(num_data_points/flip_after) + 1
    num_data_points = num_flips * flip_after
    flipping_bits = torch.randint(2, size=(num_flips, num_flipping_bits), dtype=torch.float32)
    if num_flipping_bits > 0:
        if flip_one:
            for i in range(1, num_flips):
                flipping_bits[i] = flipping_bits[i-1]
                bit_to_flip = torch.randint(num_flipping_bits, (1, ))
                flipping_bits[i][bit_to_flip] = 1 - flipping_bits[i-1][bit_to_flip]

        flipping_bits = flipping_bits.repeat_interleave(flip_after, dim=0)
        random_bits = torch.randint(2, size=(num_data_points, num_inputs - num_flipping_bits), dtype=torch.float32)

        X = torch.cat((flipping_bits, random_bits), dim=1)
    else:
        X = torch.randint(2, size=(num_data_points, num_inputs), dtype=torch.float32)

    Y = torch.zeros((num_data_points, 1), dtype=torch.float)

    with torch.no_grad():
        mini_batch_size = 10000
        for i in tqdm(range(int(num_data_points/mini_batch_size))):
            Y[i*mini_batch_size:(i+1)*mini_batch_size], features =\
                target_network.predict(x=X[i*mini_batch_size:(i+1)*mini_batch_size])

    data = X, Y, target_network
    with open(data_file, 'wb+') as f:
        pickle.dump(data, f)


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', help="Path of the file containing the parameters of the experiment",
                        type=str, default='env_temp_cfg/0.json')
    args = parser.parse_args(arguments)
    cfg_file = args.c

    with open(cfg_file, 'r') as f:
        params = json.load(f)

    if 'target_net_file' not in params.keys():
        params['target_net_file'] = None
    elif params['target_net_file'] == '':
        params['target_net_file'] = None
    if 'add_noise' not in params.keys():
        params['add_noise'] = True
    if 'flip_one' not in params.keys():
        params['flip_one'] = False

    generate_problem_data(
        data_file=params['env_file'],
        num_data_points=int(params['num_data_points']),
        flip_after=int(params['flip_after']),
        num_inputs=params['num_inputs'],
        num_target_features=params['num_target_features'],
        num_flipping_bits=params['num_flipping_bits'],
        beta=params['beta'],
        flip_one=params['flip_one'],
    )


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

