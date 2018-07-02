# encoding: utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks import ResBlock


class ResNet6n(nn.Module):
    def __init__(self, output_shape, n):

        super(ResNet6n, self).__init__()
        self.output_shape = output_shape
        self.n = n

        # regiter first conv layer and last fc layer
        self.conv_a = nn.Conv2d(3, 16, kernel_size = 3, padding = 1)
        self.bn = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64, output_shape, bias = True)
        
        # register conv blocks
        self.conv_block_b = ResBlock(16, 32, downsample = True)
        self.conv_block_c = ResBlock(32, 64, downsample = True)

        # register identity blocks
        self.iden_blocks_a = nn.ModuleList([ResBlock(16, 16) for i in range(n)])
        self.iden_blocks_b = nn.ModuleList([ResBlock(32, 32) for i in range(n-1)])
        self.iden_blocks_c = nn.ModuleList([ResBlock(64, 64) for i in range(n-1)])

        # weights init
        self.weights_init()


    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, inputs):
        # conv layer a
        x = self.conv_a(inputs)
        # expect x size 16x32x32 here
        assert(x.shape[1:] == torch.Size([16,32,32]))


        # identity blocks a
        for block in self.iden_blocks_a:
            x = block(x)

        # expect x size 16x32x32 here
        assert(x.shape[1:] == torch.Size([16,32,32]))


        # conv and identity blocks b
        x = self.conv_block_b(x)
        for block in self.iden_blocks_b:
            x = block(x)

        # expect x size 32x16x16 here
        assert(x.shape[1:] == torch.Size([32,16,16]))


        # conv and identity blocks c
        x = self.conv_block_c(x)
        for block in self.iden_blocks_c:
            x = block(x)

        # expect x size 64x8x8 here
        assert(x.shape[1:] == torch.Size([64,8,8]))


        # global pooling and fc
        x = F.relu(self.bn(x))
        x = F.avg_pool2d(x, x.shape[-1])
        x = x.view(-1, 64)
        x = self.fc(x)

        return F.softmax(x, dim = 1)


if __name__ == "__main__":
    from torchvision.datasets import CIFAR10
    import torchvision.transforms as transforms
    import numpy as np

    transform = transforms.Compose([
        transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CIFAR10("~/dataset/cifar10", transform = transform)
    x = trainset[0][0].unsqueeze(0)
    net = ResNet6n(10, n = 18)
    y = net(x)

    print(x.shape)
    print(y.shape)

    size = 1
    for param in net.parameters():
        arr = np.array(param.size())
        
        s = 1
        for e in arr:
            s *= e

        size += s

    print("all parameters %.2fM" %(size/1e6) )
