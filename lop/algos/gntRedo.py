import torch
from math import sqrt


class GnTredo(object):
    """
    Generate-and-Test algorithm for feed forward neural networks, based on ReDo
    """
    def __init__(
            self,
            net,
            hidden_activation,
            threshold=0.01,
            init='kaiming',
            device="cpu",
            reset_period=1000
    ):
        super(GnTredo, self).__init__()
        self.device = device
        self.net = net
        self.num_hidden_layers = int(len(self.net)/2)
        self.threshold = threshold
        self.steps_since_last_redo = 0
        self.reset_period = reset_period
        # Calculate uniform distribution's bound for random feature initialization
        if hidden_activation == 'selu': init = 'lecun'
        self.bounds = self.compute_bounds(hidden_activation=hidden_activation, init=init)

    def compute_bounds(self, hidden_activation, init='kaiming'):
        if hidden_activation in ['swish', 'elu']: hidden_activation = 'relu'
        if init == 'default':
            bounds = [sqrt(1 / self.net[i * 2].in_features) for i in range(self.num_hidden_layers)]
        elif init == 'xavier':
            bounds = [torch.nn.init.calculate_gain(nonlinearity=hidden_activation) *
                      sqrt(6 / (self.net[i * 2].in_features + self.net[i * 2].out_features)) for i in
                      range(self.num_hidden_layers)]
        elif init == 'lecun':
            bounds = [sqrt(3 / self.net[i * 2].in_features) for i in range(self.num_hidden_layers)]
        else:
            bounds = [torch.nn.init.calculate_gain(nonlinearity=hidden_activation) *
                      sqrt(3 / self.net[i * 2].in_features) for i in range(self.num_hidden_layers)]
        bounds.append(1 * sqrt(3 / self.net[self.num_hidden_layers * 2].in_features))
        return bounds

    def units_to_replace(self, features):
        """
        Args:
            features: Activation values in the neural network, mini-batch * layer-idx * feature-idx
        Returns:
            Features to replace in each layer, Number of features to replace in each layer
        """
        features = features.mean(dim=0)
        features_to_replace = [None]*self.num_hidden_layers
        num_features_to_replace = [None]*self.num_hidden_layers
        for i in range(self.num_hidden_layers):
            # Find features to replace
            feature_utility = features[i] / features[i].mean()
            new_features_to_replace = (feature_utility <= self.threshold).nonzero().reshape(-1)
            # Initialize utility for new features
            features_to_replace[i] = new_features_to_replace
            num_features_to_replace[i] = new_features_to_replace.shape[0]

        return features_to_replace, num_features_to_replace

    def gen_new_features(self, features_to_replace, num_features_to_replace):
        """
        Generate new features: Reset input and output weights for low utility features
        """
        with torch.no_grad():
            for i in range(self.num_hidden_layers):
                if num_features_to_replace[i] == 0:
                    continue
                current_layer = self.net[i * 2]
                next_layer = self.net[i * 2 + 2]
                current_layer.weight.data[features_to_replace[i], :] *= 0.0
                current_layer.weight.data[features_to_replace[i], :] += \
                    torch.empty(num_features_to_replace[i], current_layer.in_features).uniform_(
                        -self.bounds[i], self.bounds[i]).to(self.device)
                current_layer.bias.data[features_to_replace[i]] *= 0

                next_layer.weight.data[:, features_to_replace[i]] = 0

    def gen_and_test(self, features_history):
        """
        Perform generate-and-test
        :param features: activation of hidden units in the neural network
        """
        self.steps_since_last_redo += 1
        if self.steps_since_last_redo < self.reset_period:
            return

        features_to_replace, num_features_to_replace = self.units_to_replace(features=features_history.abs())
        self.gen_new_features(features_to_replace, num_features_to_replace)
        self.steps_since_last_redo = 0