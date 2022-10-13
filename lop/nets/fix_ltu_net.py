import torch
import torch.nn as nn
from lop.utils.ltu import LTU


class FixLTUNet(nn.Module):
    def __init__(self, num_inputs=20, num_features=80, beta=0.75):
        """
        A feed forward neural network with just one hidden layer of LTU activation
        This network is used as the date-generating (target) network in the Slowly Changing Regression problem
        :param num_inputs:
        :param num_features:
        :param beta: parameter for LTU
        """
        super(FixLTUNet, self).__init__()
        self.num_inputs = num_inputs
        self.num_features = num_features
        self.num_outputs = 1
        self.beta = beta

        # define the fully connected layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.num_inputs, self.num_features, bias=True))
        self.layers.append(nn.Linear(self.num_features, self.num_outputs, bias=True))

        # Initialize the weights of the network from {+1, -1}
        self.layers[0].weight.data = torch.randint(0, 2, (self.num_features, self.num_inputs), dtype=torch.float)*2 - 1
        self.layers[0].bias.data = torch.randint(0, 2, (self.num_features, ), dtype=torch.float)*2 - 1
        self.layers[1].weight.data = torch.randint(0, 2, (self.num_outputs, self.num_features), dtype=torch.float)*2 - 1
        self.layers[1].bias.data = torch.randint(0, 2, (self.num_outputs, ), dtype=torch.float)*2 - 1

        # Define the hidden activation
        with torch.no_grad():
            # S = number of '-1's in the input weights, (self.num_inputs - sum of weights)/2
            S = (self.num_inputs - torch.sum(self.layers[0].weight.data, dim=1) + self.layers[0].bias.data) / 2
            # threshold for each feature
            self.tau = self.beta*(self.num_inputs + 1) - S
            self.hidden_activation = LTU(tau=self.tau)

    def predict(self, x=None):
        """
        Forward pass
        :param x: inputs
        :return: output of the network
        """
        features = self.hidden_activation(self.layers[0](x))
        out = self.layers[1](features)
        return out, features
