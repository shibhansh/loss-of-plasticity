import math
import itertools
import numpy as np
from torch import nn
from tqdm import tqdm
from math import sqrt
from torch.nn import Conv2d, Linear
import torch
from scipy.linalg import svd


def net_init(net, orth=0, w_fac=1.0, b_fac=0.0):
    if orth:
        for module in net:
            if hasattr(module, 'weight'):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if hasattr(module, 'bias'):
                nn.init.constant_(module.bias, val=0)
    else:
        net[-1].weight.data.mul_(w_fac)
        if hasattr(net[-1], 'bias'):
            net[-1].bias.data.mul_(b_fac)

def fc_body(act_type, o_dim, h_dim, bias=True):
    activation = {'Tanh': nn.Tanh, 'ReLU': nn.ReLU, 'elu': nn.ELU, 'sigmoid':nn.Sigmoid}[act_type]
    module_list = nn.ModuleList()
    if len(h_dim) == 0:
        return module_list
    module_list.append(nn.Linear(o_dim, h_dim[0], bias=bias))
    module_list.append(activation())
    for i in range(len(h_dim) - 1):
        module_list.append(nn.Linear(h_dim[i], h_dim[i + 1], bias=bias))
        module_list.append(activation())
    return module_list


def get_configurations(params: {}):
    # get all parameter configurations for individual runs
    list_params = [key for key in params.keys() if type(params[key]) is list]
    param_values = [params[key] for key in list_params]
    hyper_param_settings = list(itertools.product(*param_values))
    return list_params, hyper_param_settings


def bin_m_errs(errs, m=10000):
    mses = []
    for j in tqdm(range(int(errs.shape[0]/m))):
        mses.append(errs[j*m:(j+1)*m].mean())
    return torch.tensor(mses)


def gaussian_init(net, std_dev=1e-1):
    for module in net:
        if hasattr(module, 'weight'):
            nn.init.normal_(module.weight, mean=0.0, std=std_dev)
        if hasattr(module, 'bias'):
            nn.init.normal_(module.bias, mean=0.0, std=std_dev)


def kaiming_init(net, act='relu', bias=True):
    if act == 'elu':
        act = 'relu'
    for module in net[:-1]:
        if hasattr(module, 'weight'):
            nn.init.kaiming_uniform_(module.weight, nonlinearity=act.lower())
            if bias:
                module.bias.data.fill_(0.0)
    nn.init.kaiming_uniform_(net[-1].weight, nonlinearity='linear')
    if bias:
        net[-1].bias.data.fill_(0.0)


def xavier_init(net, act='tanh', bias=True):
    if act == 'elu':
        act = 'relu'
    gain = nn.init.calculate_gain(act.lower(), param=None)
    for module in net[:-1]:
        if hasattr(module, 'weight'):
            nn.init.xavier_uniform_(module.weight, gain=gain)
            if bias:
                module.bias.data.fill_(0.0)
    nn.init.xavier_uniform_(net[-1].weight, gain=1)
    if bias:
        net[-1].bias.data.fill_(0.0)


def lecun_init(net, bias=True):
    for module in net[:-1]:
        if hasattr(module, 'weight'):
            new_bound = math.sqrt(3/module.in_features)
            nn.init.uniform_(module.weight, a=-new_bound, b=new_bound)
            if bias:
                module.bias.data.fill_(0.0)
    new_bound = math.sqrt(3/net[-1].in_features)
    nn.init.uniform_(net[-1].weight, a=-new_bound, b=new_bound)
    if bias:
        net[-1].bias.data.fill_(0.0)


def register_hook(net, hook_fn):
    for name, layer in net._modules.items():
        # If it is a sequential, don't register a hook on it but recursively register hook on all it's module children
        if isinstance(layer, nn.Sequential):
            register_hook(layer)
        else:
            # it's a non sequential. Register a hook
            layer.register_forward_hook(hook_fn)


def nll_accuracy(out, yb):
    predictions = torch.argmax(out, dim=1)
    return (predictions == yb).float().mean()


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in tqdm(range(0, inputs.shape[0], batchsize)):
        if shuffle:
            excerpt = indices[start_idx: start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def get_layer_bound(layer, init, gain):
    if isinstance(layer, Conv2d):
        return sqrt(1 / (layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]))
    elif isinstance(layer, Linear):
        if init == 'default':
            bound = sqrt(1 / layer.in_features)
        elif init == 'xavier':
            bound = gain * sqrt(6 / (layer.in_features + layer.out_features))
        elif init == 'lecun':
            bound = sqrt(3 / layer.in_features)
        else:
            bound = gain * sqrt(3 / layer.in_features)
        return bound


def compute_matrix_rank_summaries(m: torch.Tensor, prop=0.99, use_scipy=False):
    """
    Computes the rank, effective rank, and approximate rank of a matrix
    Refer to the corresponding functions for their definitions
    :param m: (float np array) a rectangular matrix
    :param prop: (float) proportion used for computing the approximate rank
    :param use_scipy: (bool) indicates whether to compute the singular values in the cpu, only matters when using
                                  a gpu
    :return: (torch int32) rank, (torch float32) effective rank, (torch int32) approximate rank
    """
    if use_scipy:
        np_m = m.cpu().numpy()
        sv = torch.tensor(svd(np_m, compute_uv=False, lapack_driver="gesvd"), device=m.device)
    else:
        sv = torch.linalg.svdvals(m)    # for large matrices, svdvals may fail to converge in gpu, but not cpu
    rank = torch.count_nonzero(sv).to(torch.int32)
    effective_rank = compute_effective_rank(sv)
    approximate_rank = compute_approximate_rank(sv, prop=prop)
    approximate_rank_abs = compute_abs_approximate_rank(sv, prop=prop)
    return rank, effective_rank, approximate_rank, approximate_rank_abs


def compute_effective_rank(sv: torch.Tensor):
    """
    Computes the effective rank as defined in this paper: https://ieeexplore.ieee.org/document/7098875/
    When computing the shannon entropy, 0 * log 0 is defined as 0
    :param sv: (float torch Tensor) an array of singular values
    :return: (float torch Tensor) the effective rank
    """
    norm_sv = sv / torch.sum(torch.abs(sv))
    entropy = torch.tensor(0.0, dtype=torch.float32, device=sv.device)
    for p in norm_sv:
        if p > 0.0:
            entropy -= p * torch.log(p)

    effective_rank = torch.tensor(np.e) ** entropy
    return effective_rank.to(torch.float32)


def compute_approximate_rank(sv: torch.Tensor, prop=0.99):
    """
    Computes the approximate rank as defined in this paper: https://arxiv.org/pdf/1909.12255.pdf
    :param sv: (float np array) an array of singular values
    :param prop: (float) proportion of the variance captured by the approximate rank
    :return: (torch int 32) approximate rank
    """
    sqrd_sv = sv ** 2
    normed_sqrd_sv = torch.flip(torch.sort(sqrd_sv / torch.sum(sqrd_sv))[0], dims=(0,))   # descending order
    cumulative_ns_sv_sum = 0.0
    approximate_rank = 0
    while cumulative_ns_sv_sum < prop:
        cumulative_ns_sv_sum += normed_sqrd_sv[approximate_rank]
        approximate_rank += 1
    return torch.tensor(approximate_rank, dtype=torch.int32)


def compute_abs_approximate_rank(sv: torch.Tensor, prop=0.99):
    """
    Computes the approximate rank as defined in this paper, just that we won't be squaring the singular values
    https://arxiv.org/pdf/1909.12255.pdf
    :param sv: (float np array) an array of singular values
    :param prop: (float) proportion of the variance captured by the approximate rank
    :return: (torch int 32) approximate rank
    """
    sqrd_sv = sv
    normed_sqrd_sv = torch.flip(torch.sort(sqrd_sv / torch.sum(sqrd_sv))[0], dims=(0,))   # descending order
    cumulative_ns_sv_sum = 0.0
    approximate_rank = 0
    while cumulative_ns_sv_sum < prop:
        cumulative_ns_sv_sum += normed_sqrd_sv[approximate_rank]
        approximate_rank += 1
    return torch.tensor(approximate_rank, dtype=torch.int32)
