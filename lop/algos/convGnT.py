from torch.nn import Conv2d, Linear
from torch import where, rand, topk, long, empty, zeros, no_grad, tensor
import torch
import sys
from lop.utils.AdamGnT import AdamGnT
from torch.nn.init import calculate_gain
from lop.utils.miscellaneous import get_layer_bound


class ConvGnT(object):
    """
    Generate-and-Test algorithm for ConvNets, maturity threshold based tester, accumulates probability of replacement,
    with various measures of feature utility
    """
    def __init__(self, net, hidden_activation, opt, decay_rate=0.99, replacement_rate=1e-4, init='kaiming',
                 num_last_filter_outputs=4, util_type='contribution', maturity_threshold=100, device='cpu'):
        super(ConvGnT, self).__init__()

        self.net = net
        self.num_hidden_layers = int(len(self.net)/2)
        self.util_type = util_type
        self.device = device

        self.opt = opt
        self.opt_type = 'sgd'
        if isinstance(self.opt, AdamGnT):
            self.opt_type = 'AdamGnT'

        """
        Define the hyper-parameters of the algorithm
        """
        self.replacement_rate = replacement_rate
        self.decay_rate = decay_rate
        self.num_last_filter_outputs = num_last_filter_outputs
        self.maturity_threshold = maturity_threshold
        self.util_type = util_type

        """
        Utility of all features/neurons
        """
        self.util, self.bias_corrected_util, self.ages, self.mean_feature_act, self.mean_abs_feature_act, \
             = [], [], [], [], []

        for i in range(self.num_hidden_layers):
            if isinstance(self.net[i * 2], Conv2d):
                self.util.append(zeros(self.net[i * 2].out_channels))
                self.bias_corrected_util.append(zeros(self.net[i * 2].out_channels))
                self.ages.append(zeros(self.net[i * 2].out_channels))
                self.mean_feature_act.append(zeros(self.net[i * 2].out_channels))
                self.mean_abs_feature_act.append(zeros(self.net[i * 2].out_channels))
            elif isinstance(self.net[i * 2], Linear):
                self.util.append(zeros(self.net[i * 2].out_features))
                self.bias_corrected_util.append(zeros(self.net[i * 2].out_features))
                self.ages.append(zeros(self.net[i * 2].out_features))
                self.mean_feature_act.append(zeros(self.net[i * 2].out_features))
                self.mean_abs_feature_act.append(zeros(self.net[i * 2].out_features))

        self.accumulated_num_features_to_replace = [0 for i in range(self.num_hidden_layers)]
        self.m = torch.nn.Softmax(dim=1)

        """
        Calculate uniform distribution's bound for random feature initialization
        """
        if hidden_activation == 'selu': init = 'lecun'
        self.bounds = self.compute_bounds(hidden_activation=hidden_activation, init=init)
        """
        Pre calculate number of features to replace per layer per update
        """
        self.num_new_features_to_replace = []
        for i in range(self.num_hidden_layers):
            with no_grad():
                if isinstance(self.net[i * 2], Linear):
                    self.num_new_features_to_replace.append(self.replacement_rate * self.net[i * 2].out_features)
                elif isinstance(self.net[i * 2], Conv2d):
                    self.num_new_features_to_replace.append(self.replacement_rate * self.net[i * 2].out_channels)

    def compute_bounds(self, hidden_activation, init='kaiming'):
        if hidden_activation in ['swish', 'elu']: hidden_activation = 'relu'
        bounds = []
        gain = calculate_gain(nonlinearity=hidden_activation)
        for i in range(self.num_hidden_layers):
            bounds.append(get_layer_bound(layer=self.net[i * 2], init=init, gain=gain))
        bounds.append(get_layer_bound(layer=self.net[-1], init=init, gain=1))
        return bounds

    def update_utility(self, layer_idx=0, features=None):
        with torch.no_grad():
            self.util[layer_idx] *= self.decay_rate
            bias_correction = 1 - self.decay_rate ** self.ages[layer_idx]

            current_layer = self.net[layer_idx * 2]
            next_layer = self.net[layer_idx * 2 + 2]

            if isinstance(next_layer, Linear):
                output_wight_mag = next_layer.weight.data.abs().mean(dim=0)
            elif isinstance(next_layer, Conv2d):
                output_wight_mag = next_layer.weight.data.abs().mean(dim=(0, 2, 3))

            self.mean_feature_act[layer_idx] *= self.decay_rate
            self.mean_abs_feature_act[layer_idx] *= self.decay_rate
            if isinstance(current_layer, Linear):
                input_wight_mag = current_layer.weight.data.abs().mean(dim=1)
                self.mean_feature_act[layer_idx] -=- (1 - self.decay_rate) * features.mean(dim=0)
                self.mean_abs_feature_act[layer_idx] -=- (1 - self.decay_rate) * features.abs().mean(dim=0)
            elif isinstance(current_layer, Conv2d):
                input_wight_mag = current_layer.weight.data.abs().mean(dim=(1, 2, 3))
                if isinstance(next_layer, Conv2d):
                    self.mean_feature_act[layer_idx] -=- (1 - self.decay_rate) * features.mean(dim=(0, 2, 3))
                    self.mean_abs_feature_act[layer_idx] -=- (1 - self.decay_rate) * features.abs().mean(dim=(0, 2, 3))
                else:
                    self.mean_feature_act[layer_idx] -=- (1 - self.decay_rate) * features.mean(dim=0).view(-1, self.num_last_filter_outputs).mean(dim=1)
                    self.mean_abs_feature_act[layer_idx] -=- (1 - self.decay_rate) * features.abs().mean(dim=0).view(-1, self.num_last_filter_outputs).mean(dim=1)

            bias_corrected_act = self.mean_feature_act[layer_idx] / bias_correction

            if self.util_type == 'adaptation':
                new_util = 1 / input_wight_mag
            elif self.util_type in ['contribution', 'zero_contribution', 'adaptable_contribution']:
                if self.util_type == 'contribution':
                    bias_corrected_act = 0
                else:
                    if isinstance(current_layer, Conv2d):
                        if isinstance(next_layer, Conv2d):
                            bias_corrected_act = bias_corrected_act.view(1, -1, 1, 1)
                        else:
                            bias_corrected_act = bias_corrected_act.repeat_interleave(self.num_last_filter_outputs).view(1, -1)
                if isinstance(next_layer, Linear):
                    if isinstance(current_layer, Linear):
                        new_util = output_wight_mag * (features - bias_corrected_act).abs().mean(dim=0)
                    elif isinstance(current_layer, Conv2d):
                        new_util = (output_wight_mag * (features - bias_corrected_act).abs().mean(dim=0)).view(-1, self.num_last_filter_outputs).mean(dim=1)
                elif isinstance(next_layer, Conv2d):
                    new_util = output_wight_mag * (features - bias_corrected_act).abs().mean(dim=(0, 2, 3))
                if self.util_type == 'adaptable_contribution':
                    new_util = new_util / input_wight_mag

            if self.util_type == 'random':
                self.bias_corrected_util[layer_idx] = rand(self.util[layer_idx].shape)
            else:
                self.util[layer_idx] -=- (1 - self.decay_rate) * new_util
                # correct the bias in the utility computation
                self.bias_corrected_util[layer_idx] = self.util[layer_idx] / bias_correction

    def test_features(self, features):
        """
        Args:
            features: Activation values in the neural network
        Returns:
            Features to replace in each layer, Number of features to replace in each layer
        """
        features_to_replace_input_indices = [empty(0, dtype=long) for _ in range(self.num_hidden_layers)]
        features_to_replace_output_indices = [empty(0, dtype=long) for _ in range(self.num_hidden_layers)]
        num_features_to_replace = [0 for _ in range(self.num_hidden_layers)]
        if self.replacement_rate == 0:
            return features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace

        for i in range(self.num_hidden_layers):
            self.ages[i] += 1
            """
            Update feature utility
            """
            self.update_utility(layer_idx=i, features=features[i])
            """
            Find the no. of features to replace
            """
            eligible_feature_indices = where(self.ages[i] > self.maturity_threshold)[0]
            if eligible_feature_indices.shape[0] == 0:
                continue
            self.accumulated_num_features_to_replace[i] -=- self.num_new_features_to_replace[i]

            """
            Case when the number of features to be replaced is between 0 and 1.
            """
            num_new_features_to_replace = int(self.accumulated_num_features_to_replace[i])
            self.accumulated_num_features_to_replace[i] -= num_new_features_to_replace

            if num_new_features_to_replace == 0:    continue

            """
            Find features to replace in the current layer
            """
            new_features_to_replace = topk(-self.bias_corrected_util[i][eligible_feature_indices],
                                           num_new_features_to_replace)[1]
            new_features_to_replace = eligible_feature_indices[new_features_to_replace]

            """
            Initialize utility for new features
            """
            self.util[i][new_features_to_replace] = 0
            self.mean_feature_act[i][new_features_to_replace] = 0.
            self.mean_abs_feature_act[i][new_features_to_replace] = 0.

            num_features_to_replace[i] = num_new_features_to_replace
            features_to_replace_input_indices[i] = new_features_to_replace
            features_to_replace_output_indices[i] = new_features_to_replace
            if isinstance(self.net[i * 2], Conv2d) and isinstance(self.net[i * 2 + 2], Linear):
                features_to_replace_output_indices[i] = \
                    (new_features_to_replace*self.num_last_filter_outputs).repeat_interleave(self.num_last_filter_outputs) + \
                    tensor([i for i in range(self.num_last_filter_outputs)]).repeat(new_features_to_replace.size()[0])

        return features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace

    def update_optim_params(self, features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace):
        """
        Update Optimizer's state
        """
        if self.opt_type == 'AdamGnT':
            for i in range(self.num_hidden_layers):
                # input weights
                if num_features_to_replace == 0:
                    continue
                # input weights
                self.opt.state[self.net[i * 2].bias]['exp_avg'][features_to_replace_input_indices[i]] = 0.0
                self.opt.state[self.net[i * 2].weight]['exp_avg_sq'][features_to_replace_input_indices[i], :] = 0.0
                self.opt.state[self.net[i * 2].bias]['exp_avg_sq'][features_to_replace_input_indices[i]] = 0.0
                self.opt.state[self.net[i * 2].weight]['step'][features_to_replace_input_indices[i], :] = 0
                self.opt.state[self.net[i * 2].bias]['step'][features_to_replace_input_indices[i]] = 0
                # output weights
                self.opt.state[self.net[i * 2 + 2].weight]['exp_avg'][:, features_to_replace_output_indices[i]] = 0.0
                self.opt.state[self.net[i * 2 + 2].weight]['exp_avg_sq'][:, features_to_replace_output_indices[i]] = 0.0
                self.opt.state[self.net[i * 2 + 2].weight]['step'][:, features_to_replace_output_indices[i]] = 0

    def gen_new_features(self, features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace):
        """
        Generate new features: Reset input and output weights for low utility features
        """
        with torch.no_grad():
            for i in range(self.num_hidden_layers):
                if num_features_to_replace[i] == 0:
                    continue
                current_layer = self.net[i * 2]
                next_layer = self.net[i * 2 + 2]

                if isinstance(current_layer, Linear):
                    current_layer.weight.data[features_to_replace_input_indices[i], :] *= 0.0
                    current_layer.weight.data[features_to_replace_input_indices[i], :] -= - \
                        empty(num_features_to_replace[i], current_layer.in_features).uniform_(-self.bounds[i],
                                                                                                self.bounds[i]).to(self.device)
                elif isinstance(current_layer, Conv2d):
                    current_layer.weight.data[features_to_replace_input_indices[i], :] *= 0.0
                    current_layer.weight.data[features_to_replace_input_indices[i], :] -= - \
                        empty([num_features_to_replace[i]] + list(current_layer.weight.shape[1:])). \
                            uniform_(-self.bounds[i], self.bounds[i])

                current_layer.bias.data[features_to_replace_input_indices[i]] *= 0.0
                """
                # Set the outgoing weights and ages to zero
                """
                next_layer.weight.data[:, features_to_replace_output_indices[i]] = 0
                self.ages[i][features_to_replace_input_indices[i]] = 0

    def gen_and_test(self, features):
        """
        Perform generate-and-test
        :param features: activation of hidden units in the neural network
        """
        if not isinstance(features, list):
            print('features passed to generate-and-test should be a list')
            sys.exit()
        features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace = self.test_features(features=features)
        self.gen_new_features(features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace)
        self.update_optim_params(features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace)
