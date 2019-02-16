import torch.nn.functional as F
import torch.nn as nn

#### Building blocks
class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


## Resnet without batchNorm
class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, k=3):
        super().__init__()
        p = 1 if k == 3 else 2

        # Conv Layer 1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(k, k), stride=stride, padding=p)

        # Conv Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=(k, k), stride=1, padding=p)

        # Shortcut connection to downsample residual
        self.shortcut = nn.Sequential()  ## equivalent to identity layer
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=(1, 1), stride=stride)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResLinear(nn.Module):
    def __init__(self, in_features, out_features, activation=nn.ReLU()):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        self.linear = nn.Linear(in_features, out_features)
        if in_features != out_features:
            self.project_linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        inner = self.activation(self.linear(x))
        if self.in_features != self.out_features:
            skip = self.project_linear(x)
        else:
            skip = x
        return inner + skip


class CIFARResNet18(nn.Module):
    def __init__(self, num_classes=10, k=3):
        super().__init__()
        self.k = k
        self.p = 1 if k == 3 else 2
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=(k, k),
            stride=1, padding=p)

        # Create stages 1-4
        self.stage1 = self._create_stage(64, 64, stride=1)
        self.stage2 = self._create_stage(64, 128, stride=2)
        self.stage3 = self._create_stage(128, 256, stride=2)
        self.stage4 = self._create_stage(256, 512, stride=2)
        self.linear = nn.Linear(2048, num_classes)
        self.flatten = nn.Sequential(Flatten())

    # A stage is just two residual blocks for ResNet18
    def _create_stage(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride, k=self.k),
            ResidualBlock(out_channels, out_channels, 1, k=self.k)
        )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = F.avg_pool2d(out, 4)
        out = self.flatten(out)
        out = self.linear(out)
        return out
