from torch import optim
from lop.algos.convGnT import ConvGnT
import torch.nn.functional as F
from lop.utils.AdamGnT import AdamGnT


class ConvCBP(object):
    """
    The Continual Backprop algorithm
    """
    def __init__(self, net, step_size=0.001, loss='mse', opt='sgd', beta=0.9, beta_2=0.999, replacement_rate=0.0001,
                 decay_rate=0.9, init='kaiming', util_type='contribution', maturity_threshold=100, device='cpu',
                 momentum=0, weight_decay=0):
        self.net = net

        # define the optimizer
        if opt == 'sgd':
            self.opt = optim.SGD(self.net.parameters(), lr=step_size, momentum=momentum, weight_decay=weight_decay)
        elif opt == 'adam':
            self.opt = AdamGnT(self.net.parameters(), lr=step_size, betas=(beta, beta_2), weight_decay=weight_decay)

        # define the loss function
        self.loss_func = {'nll': F.cross_entropy, 'mse': F.mse_loss}[loss]

        # a placeholder
        self.previous_features = None

        # define the generate-and-test object for the given network
        self.gnt = ConvGnT(
            net=self.net.layers,
            hidden_activation=self.net.act_type,
            opt=self.opt,
            replacement_rate=replacement_rate,
            decay_rate=decay_rate,
            init=init,
            num_last_filter_outputs=net.last_filter_output,
            util_type=util_type,
            maturity_threshold=maturity_threshold,
            device=device,
        )

    def learn(self, x, target):
        """
        Learn using one step of gradient-descent and generate-&-test
        :param x: input
        :param target: desired output
        :return: loss
        """
        # do a forward pass and get the hidden activations
        output, features = self.net.predict(x=x)
        loss = self.loss_func(output, target)
        self.previous_features = features

        # do the backward pass and take a gradient step
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()

        # take a generate-and-test step
        self.gnt.gen_and_test(features=self.previous_features)

        return loss.detach(), output
