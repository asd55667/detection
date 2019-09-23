import torch
from torch import nn


def dbl(in_channels, out_channels, kernel_size, stride=1, pad=0):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad),
                         nn.BatchNorm2d(out_channels),
                         nn.LeakyReLU(inplace=True))

class Darknet53(nn.Module):
    def __init__(self, layer_size=[1,2,8,8,4], in_channels=3):
        super(Darknet53, self).__init__()
        self.layer0 = dbl(in_channels, 32, 3, 1, 1)
        self.layer1 = self._make_layer(DarkBottleneck, 32,  64, 1)
        self.layer2 = self._make_layer(DarkBottleneck, 64,  128, 2)
        self.layer3 = self._make_layer(DarkBottleneck, 128, 256, 8)
        self.layer4 = self._make_layer(DarkBottleneck, 256, 512, 8)
        self.layer5 = self._make_layer(DarkBottleneck, 512, 1024, 4)
        
    def _make_layer(self, block, in_channels, out_channels, n_blocks,):
        # subsample layer
        layers = [dbl(in_channels, out_channels, 3, 2, 1)]
        for i in range(n_blocks):
            layers.append(block(out_channels, in_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        C1 = self.layer1(x)
        C2 = self.layer2(C1)
        C3 = self.layer3(C2)
        C4 = self.layer4(C3)
        C5 = self.layer5(C4)
        return C3, C4, C5


class Bottleneck(nn.Module):

    def forward(self, x):
        
        preact = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        residual = x + preact
        return self.relu(residual)

class DarkBottleneck(Bottleneck):

    def __init__(self, in_channels, out_channels):
        super(DarkBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1,)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, in_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU(inplace=True) 

# import collections
# class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
#     'A named tuple describing a Draknet53 block'
#     block = [Block('C1', Bottleneck, [32, 64,  2]),
#                 Block('C2', Bottleneck, [64, 128, 2] + [64, 128, 1]),
#                 Block('C3', Bottleneck, [128, 256, 2] +  [128, 256, 1] * 7),
#                 Block('C4', Bottleneck, [256, 512, 2] +  [256, 512, 1] * 7),
#                 Block('C5', Bottleneck, [512, 1024, 2] + [512, 1024, 1] * 3)]

if __name__ == '__main__':
    from torchsummary import summary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Darknet53().to(device)
    summary(model, (3, 416,416))