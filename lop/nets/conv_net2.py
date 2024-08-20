import torch.nn as nn
from lop.algos.cbp_linear import CBPLinear
from lop.algos.cbp_conv import CBPConv


class ConvNet2(nn.Module):
    def __init__(self, num_classes=10, replacement_rate=0, init='default', maturity_threshold=100):

        """
        Same as ConvNet, but using CBP-layers
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.last_filter_output = 2 * 2
        self.num_conv_outputs = 128 * self.last_filter_output
        self.fc1 = nn.Linear(self.num_conv_outputs, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
        self.act = nn.ReLU()

        """
        Initialize CBP-layers
        """
        self.cbp1 = CBPConv(in_layer=self.conv1, out_layer=self.conv2, replacement_rate=replacement_rate, maturity_threshold=maturity_threshold, init=init)
        self.cbp2 = CBPConv(in_layer=self.conv2, out_layer=self.conv3, replacement_rate=replacement_rate, maturity_threshold=maturity_threshold, init=init)
        self.cbp3 = CBPConv(in_layer=self.conv3, out_layer=self.fc1, num_last_filter_outputs=self.last_filter_output, replacement_rate=replacement_rate, maturity_threshold=maturity_threshold, init=init)
        self.cbp4 = CBPLinear(in_layer=self.fc1, out_layer=self.fc2, replacement_rate=replacement_rate, maturity_threshold=maturity_threshold, init=init)
        self.cbp5 = CBPLinear(in_layer=self.fc2, out_layer=self.fc3, replacement_rate=replacement_rate, maturity_threshold=maturity_threshold, init=init)

        self.layers = nn.ModuleList()
        self.layers.append(self.conv1)
        self.layers.append(nn.ReLU())
        self.layers.append(self.conv2)
        self.layers.append(nn.ReLU())
        self.layers.append(self.conv3)
        self.layers.append(nn.ReLU())
        self.layers.append(self.fc1)
        self.layers.append(nn.ReLU())
        self.layers.append(self.fc2)
        self.layers.append(nn.ReLU())
        self.layers.append(self.fc3)

        self.act_type = 'relu'

    def predict(self, x):
        """
        The input passes through CBP layers after the non-linearities
        """
        x1 = self.cbp1(self.pool(self.act(self.conv1(x))))
        x2 = self.cbp2(self.pool(self.act(self.conv2(x1))))
        x3 = self.cbp3(self.pool(self.act(self.conv3(x2))))
        x3 = x3.view(-1, self.num_conv_outputs)
        x4 = self.cbp4(self.act(self.fc1(x3)))
        x5 = self.cbp5(self.act(self.fc2(x4)))
        x6 = self.fc3(x5)

        return x6, [x1, x2, x3, x4, x5]
