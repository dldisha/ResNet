import torch
import torch.nn as nn


# N:    residual layer (default: 4)
# B_i:  residual block in layer i (default: 2)
# C_1:  channel in layer 1 (default: 64)
# F_i:  filter in residual conv (default: 3)
# K_i:  filter in connection conv (default: 1)
# P:    average pooling size (default: 1)


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
        self.conv2 = convNxN(C_in=C_in, C_out=C_out, F=F, stride=1, padding=(F - 1) // 2)
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
            norm_layer=nn.BatchNorm2d
    ) -> None:
        super(ResNet, self).__init__()
        self.inplanes = C_1
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        # Input Layer
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=F[0], stride=2, padding=(F[0] - 1) // 2, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


if __name__ == '__main__':
    model = torchvision.models.resnet18()
    print(model)
