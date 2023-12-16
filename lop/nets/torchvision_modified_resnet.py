from typing import Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

from torchvision.utils import _log_api_usage_once

""" 
This is a modified version of torchvision's code for instantiating resnets. Here's a list of the changes made to the 
source code:
    - All convolutional layers now have bias set to True, where they were original set to False.
    - Removed the first maxpool layers so that input stays somewhat large.
    - Layer conv1 in the ResNet class has kernel size set to 3 and stride set to 1, where they were originally 7 and 2,
      respectively. 
    - Forward calls have a feature list argument to store the features of the network. This is only used for continual 
      backprop and doesn't affect the output of the network.
To see the source code, go to: torchvision.models.resnet (for torchvision==0.15.1)
"""


class SequentialWithKeywordArguments(torch.nn.Sequential):

    """
    Sequential module that allows the use of keyword arguments in the forward pass
    """

    def forward(self, input, **kwargs):
        for module in self:
            input = module(input, **kwargs)
        return input


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding from torchvision.models.resnet but bias is set to True"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=True,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor, feature_list: list = None) -> Tensor:
        """
        Forward pass through the block
        :param x: input to the resnet block
        :param feature_list: list to store the features of the network
        :return: output of the block
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if feature_list is not None: feature_list.append(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        if feature_list is not None: feature_list.append(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.output_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> SequentialWithKeywordArguments:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = SequentialWithKeywordArguments(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return SequentialWithKeywordArguments(*layers)

    def _forward_impl(self, x: Tensor, feature_list: list = None) -> Tensor:
        """
        Forward pass for a resnet
        :param x: input to the network
        :param feature_list: optional list for storing the features of the network
        :return: output of the network
        """
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if feature_list is not None: feature_list.append(x)

        x = self.layer1(x, feature_list=feature_list)
        x = self.layer2(x, feature_list=feature_list)
        x = self.layer3(x, feature_list=feature_list)
        x = self.layer4(x, feature_list=feature_list)

        if feature_list is not None: feature_list.pop(-1)

        x = self.output_pool(x)
        x = torch.flatten(x, 1)

        if feature_list is not None: feature_list.append(x)

        x = self.fc(x)

        return x

    def forward(self, x: Tensor, feature_list: list = None) -> Tensor:
        return self._forward_impl(x, feature_list)


def build_resnet18(num_classes: int, norm_layer):
    """
    :param num_classes: number of classes for the classification problem
    :param norm_layer: type of normalization. Options: [torch.nn.BatchNorm2d, torch.nn.Identity]
    :return: an instance of ResNet with the correct number of layers for ResNet34
    """
    return ResNet(BasicBlock, layers=[2, 2, 2, 2], norm_layer=norm_layer, num_classes=num_classes)


def kaiming_init_resnet_module(nn_module: torch.nn.Module):
    """
    Initializes the parameters of a resnet module in the following way:
        - Conv2d: weights are initialized using xavier normal initialization and bias are initialized to zero
        - Linear: same as Conv2d
        - BatchNorm2d: bias are initialized to 0, weights are initialized to 1
    :param nn_module: an instance ot torch.nn.Module to be initialized
    """

    if isinstance(nn_module, torch.nn.Conv2d) or isinstance(nn_module, torch.nn.Linear):
        if isinstance(nn_module, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(nn_module.weight, nonlinearity="relu")
        else:   # the only linear layer in a resnet is the output layer
            torch.nn.init.kaiming_normal_(nn_module.weight, nonlinearity="linear")
        if nn_module.bias is not None:
            torch.nn.init.constant_(nn_module.bias, 0.0)

    if isinstance(nn_module, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(nn_module.weight, 1.0)
        torch.nn.init.constant_(nn_module.bias, 0.0)