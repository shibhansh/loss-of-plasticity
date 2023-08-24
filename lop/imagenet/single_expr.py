import sys
import json
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from lop.algos.bp import Backprop
from lop.nets.conv_net import ConvNet
from lop.algos.convCBP import ConvCBP
from torch.nn.functional import softmax
from lop.nets.linear import MyLinear
from lop.utils.miscellaneous import nll_accuracy as accuracy

train_images_per_class = 600
test_images_per_class = 100
images_per_class = train_images_per_class + test_images_per_class


def load_imagenet(classes=[]):
    x_train, y_train, x_test, y_test = [], [], [], []
    for idx, _class in enumerate(classes):
        data_file = 'data/classes/' + str(_class) + '.npy'
        new_x = np.load(data_file)
        x_train.append(new_x[:train_images_per_class])
        x_test.append(new_x[train_images_per_class:])
        y_train.append(np.array([idx] * train_images_per_class))
        y_test.append(np.array([idx] * test_images_per_class))
    x_train = torch.tensor(np.concatenate(x_train))
    y_train = torch.from_numpy(np.concatenate(y_train))
    x_test = torch.tensor(np.concatenate(x_test))
    y_test = torch.from_numpy(np.concatenate(y_test))
    return x_train, y_train, x_test, y_test


def save_data(data, data_file):
    with open(data_file, 'wb+') as f:
        pickle.dump(data, f)


def repeat_expr(params: {}):
    agent_type = params['agent']
    num_tasks = params['num_tasks']
    num_showings = params['num_showings']

    step_size = params['step_size']
    replacement_rate = 0.0001
    decay_rate = 0.99
    maturity_threshold = 100
    util_type = 'contribution'
    opt = params['opt']
    weight_decay = 0
    use_gpu = 0
    dev='cpu'
    num_classes = 10
    total_classes = 1000
    new_heads = True
    mini_batch_size = 100
    perturb_scale = 0
    momentum = 0
    if 'replacement_rate' in params.keys(): replacement_rate = params['replacement_rate']
    if 'decay_rate' in params.keys(): decay_rate = params['decay_rate']
    if 'util_type' in params.keys(): util_type = params['util_type']
    if 'maturity_threshold' in params.keys():   maturity_threshold = params['maturity_threshold']
    if 'weight_decay' in params.keys(): weight_decay = params['weight_decay']
    if 'use_gpu' in params.keys():
        if params['use_gpu'] == 1:
            use_gpu = 1
            dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            if dev == torch.device("cuda"):    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if 'num_classes' in params.keys():  num_classes = params['num_classes']
    if 'new_heads' in params.keys():    new_heads = params['new_heads']
    if 'mini_batch_size' in params.keys():  mini_batch_size = params['mini_batch_size']
    if 'perturb_scale' in params.keys():    perturb_scale = params['perturb_scale']
    if 'momentum' in params.keys(): momentum = params['momentum']
    num_epochs = num_showings

    classes_per_task = num_classes
    net = ConvNet()
    if agent_type == 'linear':
        net = MyLinear( 
            input_size=3072, num_outputs=classes_per_task
        )

    if agent_type in ['bp', 'linear']:
        learner = Backprop(
            net=net,
            step_size=step_size,
            opt=opt,
            loss='nll',
            weight_decay=weight_decay,
            to_perturb=(perturb_scale != 0),
            perturb_scale=perturb_scale,
            device=dev,
            momentum=momentum,
        )
    elif agent_type == 'cbp':
        learner = ConvCBP(
            net=net,
            step_size=step_size,
            momentum=momentum,
            loss='nll',
            weight_decay=weight_decay,
            opt=opt,
            init='default',
            replacement_rate=replacement_rate,
            decay_rate=decay_rate,
            util_type=util_type,
            device=dev,
            maturity_threshold=maturity_threshold,
        )

    with open('class_order', 'rb+') as f:
        class_order = pickle.load(f)
        class_order = class_order[int([params['run_idx']][0])]
    num_class_repetitions_required = int(num_classes * num_tasks / total_classes) + 1
    class_order = np.concatenate([class_order]*num_class_repetitions_required)
    save_after_every_n_tasks = 1
    if num_tasks >= 10:
        save_after_every_n_tasks = int(num_tasks/10)

    examples_per_epoch = train_images_per_class * classes_per_task

    train_accuracies = torch.zeros((num_tasks, num_epochs), dtype=torch.float)
    test_accuracies = torch.zeros((num_tasks, num_epochs), dtype=torch.float)

    x_train, x_test, y_train, y_test = None, None, None, None
    for task_idx in range(num_tasks):
        del x_train, x_test, y_train, y_test
        x_train, y_train, x_test, y_test = load_imagenet(class_order[task_idx*classes_per_task:(task_idx+1)*classes_per_task])
        x_train, x_test = x_train.type(torch.FloatTensor), x_test.type(torch.FloatTensor)
        if agent_type == 'linear':
            x_train, x_test = x_train.flatten(1), x_test.flatten(1)
        if use_gpu == 1:
            x_train, x_test, y_train, y_test = x_train.to(dev), x_test.to(dev), y_train.to(dev), y_test.to(dev)
        if new_heads:
            net.layers[-1].weight.data *= 0
            net.layers[-1].bias.data *= 0

        for epoch_idx in tqdm(range(num_epochs)):
            example_order = np.random.permutation(train_images_per_class * classes_per_task)
            x_train = x_train[example_order]
            y_train = y_train[example_order]
            new_train_accuracies = torch.zeros((int(examples_per_epoch/mini_batch_size),), dtype=torch.float)
            epoch_iter = 0
            for start_idx in range(0, examples_per_epoch, mini_batch_size):
                batch_x = x_train[start_idx: start_idx+mini_batch_size]
                batch_y = y_train[start_idx: start_idx+mini_batch_size]

                # train the network
                loss, network_output = learner.learn(x=batch_x, target=batch_y)
                with torch.no_grad():
                    new_train_accuracies[epoch_iter] = accuracy(softmax(network_output, dim=1), batch_y).cpu()
                    epoch_iter += 1

            # log accuracy
            with torch.no_grad():
                train_accuracies[task_idx][epoch_idx] = new_train_accuracies.mean()
                new_test_accuracies = torch.zeros((int(x_test.shape[0] / mini_batch_size),), dtype=torch.float)
                test_epoch_iter = 0
                for start_idx in range(0, x_test.shape[0], mini_batch_size):
                    test_batch_x = x_test[start_idx: start_idx + mini_batch_size]
                    test_batch_y = y_test[start_idx: start_idx + mini_batch_size]

                    network_output, _ = net.predict(x=test_batch_x)
                    new_test_accuracies[test_epoch_iter] = accuracy(softmax(network_output, dim=1), test_batch_y)
                    test_epoch_iter += 1

                test_accuracies[task_idx][epoch_idx] = new_test_accuracies.mean()
                print('accuracy for task', task_idx, 'in epoch', epoch_idx, ': train, ',
                      train_accuracies[task_idx][epoch_idx], ', test,', test_accuracies[task_idx][epoch_idx])

        if task_idx % save_after_every_n_tasks == 0:
            save_data(data={
                'train_accuracies': train_accuracies.cpu(),
                'test_accuracies': test_accuracies.cpu(),
            }, data_file=params['data_file'])
    
    save_data(data={
        'train_accuracies': train_accuracies.cpu(),
        'test_accuracies': test_accuracies.cpu(),
    }, data_file=params['data_file'])


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

    repeat_expr(params)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
