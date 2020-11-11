from torch import nn


def conv_block(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


class Conv4(nn.Module):
    def __init__(self, num_classes, drop_dim=True, extract=False):
        super(Conv4, self).__init__()
        self.drop_dim = drop_dim
        self.extract = extract
        self.conv1 = conv_block(3, 64)
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64)
        self.conv4 = conv_block(64, 64)
        self.linear = nn.Linear(1600, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x_f = self.conv4(x)

        if self.drop_dim:
            x_f = x_f.view(x_f.size(0), -1)
        x_out = self.linear(x_f)

        if self.extract:
            return x_out, x_f
        else:
            return x_out
