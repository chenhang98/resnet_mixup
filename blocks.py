import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, input_channel, output_channel, downsample = False):
        super(ResBlock, self).__init__()
        self.downsample = downsample
        stride = 2 if downsample else 1

        # for main path
        self.bn1 = nn.BatchNorm2d(input_channel)
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size = 3, padding = 1, stride = stride)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size = 3, padding = 1)

        # for shortcut
        if self.downsample:
            self.conv3 = nn.Conv2d(input_channel, output_channel, kernel_size = 1, stride = stride)


    def forward(self, inputs):
        # two conv layers path
        x = self.conv1(F.relu(self.bn1(inputs)))
        x = self.conv2(F.relu(self.bn2(x)))

        # shortcut path
        shortcut = self.conv3(inputs) if self.downsample else inputs

        # merge two paths
        assert x.shape == shortcut.shape, "merge failed in resblock"
        return x + shortcut


if __name__ == "__main__":
    from torchvision.datasets import CIFAR10
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CIFAR10("~/dataset/cifar10", transform = transform)
    x = trainset[0][0].unsqueeze(0)

    print(x.shape)
    b = ResBlock(3, 2, True)
    y = b(x)
    print(b)
    print(y.shape, y.max(), y.min())