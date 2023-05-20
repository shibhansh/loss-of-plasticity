import sys
import json
import pickle
import argparse
from lop.nets.ffnn import FFNN
from lop.nets.linear import MyLinear
from lop.algos.bp import Backprop
from lop.algos.cbp import ContinualBackprop
from lop.utils.miscellaneous import *


def expr(params: {}):
    agent_type = params['agent']
    env_file = params['env_file']
    num_data_points = int(params['num_data_points'])
    to_log = False
    to_log_grad = False
    to_log_activation = False
    beta_1 = 0.9
    beta_2 = 0.999
    weight_decay = 0.0
    accumulate = False
    perturb_scale = 0
    if 'to_log' in params.keys():
        to_log = params['to_log']
    if 'to_log_grad' in params.keys():
        to_log_grad = params['to_log_grad']
    if 'to_log_activation' in params.keys():
        to_log_activation = params['to_log_activation']
    if 'beta_1' in params.keys():
        beta_1 = params['beta_1']
    if 'beta_2' in params.keys():
        beta_2 = params['beta_2']
    if 'weight_decay' in params.keys():
        weight_decay = params['weight_decay']
    if 'accumulate' in params.keys():
        accumulate = params['accumulate']
    if 'perturb_scale' in params.keys():
        perturb_scale = params['perturb_scale']

    num_inputs = params['num_inputs']
    num_features = params['num_features']
    hidden_activation = params['hidden_activation']
    step_size = params['step_size']
    opt = params['opt']
    replacement_rate = params["replacement_rate"]
    decay_rate = params["decay_rate"]
    mt = 10
    util_type='adaptable_contribution'
    init = 'kaiming'
    if "mt" in params.keys():
        mt = params["mt"]
    if "util_type" in params.keys():
        util_type = params["util_type"]
    if "init" in params.keys():
        init = params["init"]

    if agent_type == 'linear':
        net = MyLinear(
            input_size=num_inputs,
        )
    else:
        net = FFNN(
            input_size=num_inputs,
            num_features=num_features,
            hidden_activation=hidden_activation,
        )

    if agent_type == 'bp' or agent_type == 'linear' or agent_type == 'l2':
        learner = Backprop(
            net=net,
            step_size=step_size,
            opt=opt,
            beta_1=beta_1,
            beta_2=beta_2,
            weight_decay=weight_decay,
            to_perturb=(perturb_scale > 0),
            perturb_scale=perturb_scale,
        )
    elif agent_type == 'cbp':
        learner = ContinualBackprop(
            net=net,
            step_size=step_size,
            opt=opt,
            replacement_rate=replacement_rate,
            decay_rate=decay_rate,
            device='cpu',
            maturity_threshold=mt,
            util_type=util_type,
            init=init,
            accumulate=accumulate,
        )

    with open(env_file, 'rb+') as f:
        inputs, outputs, _ = pickle.load(f)

    errs = torch.zeros((num_data_points), dtype=torch.float)
    if to_log: weight_mag = torch.zeros((num_data_points, 2), dtype=torch.float)
    if to_log_grad: grad_mag = torch.zeros((num_data_points, 2), dtype=torch.float)
    if to_log_activation: activation = torch.zeros((num_data_points, ), dtype=torch.float)
    for i in tqdm(range(num_data_points)):
        x, y = inputs[i: i+1], outputs[i: i+1]
        err = learner.learn(x=x, target=y)
        if to_log:
            weight_mag[i][0] = learner.net.layers[0].weight.data.abs().mean()
            weight_mag[i][1] = learner.net.layers[-1].weight.data.abs().mean()
        if to_log_grad:
            grad_mag[i][0] = learner.net.layers[0].weight.grad.data.abs().mean()
            grad_mag[i][1] = learner.net.layers[-1].weight.grad.data.abs().mean()
        if to_log_activation:
            if hidden_activation == 'relu':
                activation[i] = (learner.previous_features[0] == 0).float().mean()
            if hidden_activation == 'tanh':
                activation[i] = (learner.previous_features[0].abs() > 0.9).float().mean()
        errs[i] = err

    data_to_save = {
        'errs': errs.numpy()
    }
    if to_log:
        data_to_save['weight_mag'] = weight_mag.numpy()
    if to_log_grad:
        data_to_save['grad_mag'] = grad_mag.numpy()
    if to_log_activation:
        data_to_save['activation'] = activation.numpy()
    return data_to_save


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', help="Path to the file containing the parameters for the experiment",
                        type=str, default='temp_cfg/0.json')
    args = parser.parse_args(arguments)
    cfg_file = args.c

    with open(cfg_file, 'r') as f:
        params = json.load(f)

    data = expr(params)

    with open(params['data_file'], 'wb+') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
