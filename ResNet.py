import torch
import torch.nn.functional as F
import torch.nn as nn
from prettytable import PrettyTable


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


# N:    residual layer (default: 4)
# B_i:  residual block in layer i (default: 2)
# C_1:  channel in layer 1 (default: 64)
# F_i:  filter in residual conv (default: 3)
# K_i:  filter in connection conv (default: 1)
# P:    average pooling size (default: 1)


class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)


def convNxN(
        C_in,
        C_out,
        F,
        stride=1,
        groups=1,
        dilation=1,
        padding=1
) -> nn.Conv2d:
    """NxN convolution with padding"""
    return nn.Conv2d(C_in, C_out, kernel_size=F, stride=stride,
                     padding=padding, groups=groups, bias=False, dilation=dilation)


class ResBlock(nn.Module):
    def __init__(
            self,
            C_in,
            C_out,
            F,
            stride=1,
            groups=1,
            dilation=1,
            base_width=64,
            downsample=None,
            norm_layer=nn.BatchNorm2d
    ) -> None:
        super(ResBlock, self).__init__()
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = convNxN(C_in=C_in, C_out=C_out, F=F, stride=stride, padding=(F - 1) // 2)
        self.bn1 = norm_layer(C_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = convNxN(C_in=C_out, C_out=C_out, F=F, stride=1, padding=(F - 1) // 2)
        self.bn2 = norm_layer(C_out)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
            self,
            C_1,
            N=4,
            F=[3, 3, 3, 3],
            B=[2, 2, 2, 2],
            K=[1, 1, 1, 1],
            P=1,
            num_classes=10,
            groups=1,
            width_per_group=64,
            norm_layer=nn.BatchNorm2d,
            D_P=0.,
            D_S=0
    ) -> None:
        super(ResNet, self).__init__()
        self._norm_layer = norm_layer
        self.inplanes = C_1
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        # Input Layer
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=F[0], stride=2, padding=(F[0] - 1) // 2, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(self.inplanes, self.inplanes,
                                       K[0], F[0], B[0], dropout_prob=D_P, dropout_size=D_S)
        layer2 = []
        for i in range(N - 1):
            layer2.append(self._make_layer(self.inplanes * (2 ** i),
                                           self.inplanes * (2 ** (i + 1)),
                                           K[i], F[i], B[i], stride=2,
                                           dropout_prob=D_P, dropout_size=D_S))
        self.layer2 = nn.Sequential(*layer2)
        self.avgpool = nn.AdaptiveAvgPool2d((P, P))
        final_input = int(self.inplanes * (2 ** (N - 1)) * (P ** 2))
        self.fc = nn.Linear(final_input, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, inplanes, planes, K, F, blocks, dropout_prob=0, dropout_size=0, stride=1):
        if inplanes == planes:
            downsample = None
        else:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=K, stride=stride, padding=(K - 1) // 2, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(ResBlock(inplanes, planes, F, stride, downsample=downsample))
        for _ in range(1, blocks):
            layers.append(ResBlock(planes, planes, F, downsample=None))
        if dropout_prob > 0:
            layers.append(DropBlock2D(dropout_prob, dropout_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        for layer in self.layer2:
            x = layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    model = ResNet(64, N=3, F=[3, 3, 3, 3], B=[2, 2, 2, 2], K=[1, 1, 1, 1], P=1, D_P=0.2, D_S=7)
    images = torch.randn(10, 3, 32, 32)
    print(model)
    print(count_parameters(model) / 1000 / 1000)
    print(images.size())
    outputs = model(images)
    print(outputs.size())
