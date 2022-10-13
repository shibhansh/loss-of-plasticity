import torch
from torch import nn


class ThresholdFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, tau):
        output = torch.where(inp > tau, torch.ones(tau.size()[0]), torch.zeros(tau.size()[0]))
        ctx.save_for_backward(inp, torch.tensor(tau))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inp, thresholds = ctx.saved_tensors
        sigmoid_val = torch.sigmoid(inp-thresholds)
        return grad_output * sigmoid_val * (1 - sigmoid_val), None, None


class LTU(nn.Module):
    def __init__(self, tau: torch.tensor):
        super().__init__()
        self.tau = tau
        self.func = ThresholdFunction.apply

    def forward(self, input):
        output = self.func(input, self.tau)
        return output
