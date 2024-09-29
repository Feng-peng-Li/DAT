import torch
import torch.nn as nn
import torch.nn.functional as F






class BasicBlocks(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlocks, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottlenecks(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottlenecks, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNets(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNets, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu=nn.ReLU()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def get_grad_cam_target_layer(self,mask_layer):
        if mask_layer=='1':
            return self.layer1[-1]
        elif mask_layer=='2':
            return self.layer2[-1]
        elif mask_layer=='3':
            return self.layer3[-1]
        elif mask_layer=='4':
            return self.layer4[-1]
    def forward(self, x,train=False):

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        flaten = F.avg_pool2d(out, 4)

        out = flaten.view(flaten.size(0), -1)

        out = self.linear(out)
        if train:
            return out,flaten.view(flaten.size(0), -1)
        else:
            return out


def ResNet18s(num_classes=10):
    return ResNets(BasicBlocks, [2, 2, 2, 2],num_classes=10)


def ResNet34s():
    return ResNets(BasicBlocks, [3, 4, 6, 3])


def ResNet50s():
    return ResNets(Bottlenecks, [3, 4, 6, 3])


def ResNet101s():
    return ResNets(Bottlenecks, [3, 4, 23, 3])


def ResNet152s():
    return ResNets(Bottlenecks, [3, 8, 36, 3])


def test():
    net = ResNet18s()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
