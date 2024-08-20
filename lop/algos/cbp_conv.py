import torch
from torch import nn
from torch.nn.init import calculate_gain
from lop.algos.cbp_linear import call_reinit, log_features, get_layer_bound


class CBPConv(nn.Module):
    def __init__(
            self,
            in_layer: nn.Conv2d,
            out_layer: [nn.Conv2d, nn.Linear],
            ln_layer: nn.LayerNorm = None,
            bn_layer: nn.BatchNorm2d = None,
            num_last_filter_outputs=1,
            replacement_rate=1e-5,
            maturity_threshold=1000,
            init='kaiming',
            act_type='relu',
            util_type='contribution',
            decay_rate=0,
    ):
        super().__init__()
        if type(in_layer) is not nn.Conv2d:
            raise Warning("Make sure in_layer is a convolutional layer")
        if type(out_layer) not in [nn.Linear, nn.Conv2d]:
            raise Warning("Make sure out_layer is a convolutional or linear layer")

        """
        Define the hyper-parameters of the algorithm
        """
        self.replacement_rate = replacement_rate
        self.maturity_threshold = maturity_threshold
        self.util_type = util_type
        self.decay_rate = decay_rate
        self.features = None
        self.num_last_filter_outputs = num_last_filter_outputs

        """
        Register hooks
        """
        if self.replacement_rate > 0:
            self.register_full_backward_hook(call_reinit)
            self.register_forward_hook(log_features)

        self.in_layer = in_layer
        self.out_layer = out_layer
        self.ln_layer = ln_layer
        self.bn_layer = bn_layer
        """
        Utility of all features/neurons
        """
        self.util = nn.Parameter(torch.zeros(self.in_layer.out_channels), requires_grad=False)
        self.ages = nn.Parameter(torch.zeros(self.in_layer.out_channels), requires_grad=False)
        self.accumulated_num_features_to_replace = nn.Parameter(torch.zeros(1), requires_grad=False)
        """
        Calculate uniform distribution's bound for random feature initialization
        """
        self.bound = get_layer_bound(layer=self.in_layer, init=init, gain=calculate_gain(nonlinearity=act_type))

    def forward(self, _input):
        return _input

    def get_features_to_reinit(self):
        """
        Returns: Features to replace
        """
        features_to_replace_input_indices = torch.empty(0, dtype=torch.long, device=self.util.device)
        features_to_replace_output_indices = torch.empty(0, dtype=torch.long, device=self.util.device)
        self.ages += 1
        """
        Calculate number of features to replace
        """
        eligible_feature_indices = torch.where(self.ages > self.maturity_threshold)[0]
        if eligible_feature_indices.shape[0] == 0:  return features_to_replace_input_indices, features_to_replace_output_indices

        num_new_features_to_replace = self.replacement_rate*eligible_feature_indices.shape[0]
        self.accumulated_num_features_to_replace += num_new_features_to_replace
        if self.accumulated_num_features_to_replace < 1:    return features_to_replace_input_indices, features_to_replace_output_indices

        num_new_features_to_replace = int(self.accumulated_num_features_to_replace)
        self.accumulated_num_features_to_replace -= num_new_features_to_replace
        """
        Calculate feature utility
        """
        if isinstance(self.out_layer, torch.nn.Linear):
            output_weight_mag = self.out_layer.weight.data.abs().mean(dim=0).view(-1, self.num_last_filter_outputs)
            self.util.data = (output_weight_mag * self.features.abs().mean(dim=0).view(-1, self.num_last_filter_outputs)).mean(dim=1)
        elif isinstance(self.out_layer, torch.nn.Conv2d):
            output_weight_mag = self.out_layer.weight.data.abs().mean(dim=(0, 2, 3))
            self.util.data = output_weight_mag * self.features.abs().mean(dim=(0, 2, 3))
        """
        Find features with smallest utility
        """
        new_features_to_replace = torch.topk(-self.util[eligible_feature_indices], num_new_features_to_replace)[1]
        new_features_to_replace = eligible_feature_indices[new_features_to_replace]
        features_to_replace_input_indices, features_to_replace_output_indices = new_features_to_replace, new_features_to_replace

        if isinstance(self.in_layer, torch.nn.Conv2d) and isinstance(self.out_layer, torch.nn.Linear):
            features_to_replace_output_indices = (
                    (new_features_to_replace * self.num_last_filter_outputs).repeat_interleave(self.num_last_filter_outputs) +
                    torch.tensor([i for i in range(self.num_last_filter_outputs)]).repeat(new_features_to_replace.size()[0]))
        return features_to_replace_input_indices, features_to_replace_output_indices

    def reinit_features(self, features_to_replace_input_indices, features_to_replace_output_indices):
        """
        Reset input and output weights for low utility features
        """
        with torch.no_grad():
            num_features_to_replace = features_to_replace_input_indices.shape[0]
            if num_features_to_replace == 0: return
            self.in_layer.weight.data[features_to_replace_input_indices, :] *= 0.0
            # noinspection PyArgumentList
            self.in_layer.weight.data[features_to_replace_input_indices, :] += \
                torch.empty([num_features_to_replace] + list(self.in_layer.weight.shape[1:]), device=self.util.device).uniform_(-self.bound, self.bound)
            self.in_layer.bias.data[features_to_replace_input_indices] *= 0

            self.out_layer.weight.data[:, features_to_replace_output_indices] = 0
            self.ages[features_to_replace_input_indices] = 0

            """
            Reset the corresponding batchnorm/layernorm layers
            """
            if self.bn_layer is not None:
                self.bn_layer.bias.data[features_to_replace_input_indices] = 0.0
                self.bn_layer.weight.data[features_to_replace_input_indices] = 1.0
                self.bn_layer.running_mean.data[features_to_replace_input_indices] = 0.0
                self.bn_layer.running_var.data[features_to_replace_input_indices] = 1.0
            if self.ln_layer is not None:
                self.ln_layer.bias.data[features_to_replace_input_indices] = 0.0
                self.ln_layer.weight.data[features_to_replace_input_indices] = 1.0

    def reinit(self):
        """
        Perform selective reinitialization
        """
        features_to_replace_input_indices, features_to_replace_output_indices = self.get_features_to_reinit()
        self.reinit_features(features_to_replace_input_indices, features_to_replace_output_indices)
