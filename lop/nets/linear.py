import torch.nn as nn


class MyLinear(nn.Module):
    def __init__(self, input_size, num_outputs=1):
        super(MyLinear, self).__init__()
        """
        A linear network
        """
        # initialize the weights
        self.fc1 = nn.Linear(input_size, num_outputs)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='linear')
        self.fc1.bias.data.fill_(0.0)
        self.layers = [self.fc1]

    def predict(self, x):
        """
        Forward pass
        :param x: input
        :return: estimate
        """
        out = self.fc1(x)
        return out, None
