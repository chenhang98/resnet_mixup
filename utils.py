import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def onehot(y, nb = 10):
    n = y.shape[0]
    y = y.reshape((n, 1))

    y_onehot = torch.FloatTensor(n, nb)
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)

    return y_onehot


def mixup(data1, data2, alpha):
    # require data1, data2 have shape ([n, c, w, h], [n])
    x1, y1 = data1
    x2, y2 = data2
    n, c, w, h = x1.shape

    y1 = onehot(y1)
    y2 = onehot(y2)

    lamb = np.random.beta(alpha, alpha, size = (n,1))
    lamb = np.array(lamb, dtype = "float32")

    lambx = lamb.reshape((n,1,1,1)) * np.ones_like(x1)
    lamby = lamb * np.ones_like(y1)

    x = lambx * x1 + (1 - lambx) * x2
    y = lamby * y1 + (1 - lamby) * y2

    return (x, y)



def test(model, testloader, criterion):
    # test model on testloader
    # return val_loss, val_acc
    
    model.eval()
    correct, total = 0, 0
    loss, counter = 0, 0
    
    with torch.no_grad():
        for (images, labels) in testloader:
            labels = onehot(labels)
            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)
            predicted = outputs.data.argmax(dim = 1)

            total += labels.size(0)
            # print(predicted.shape, labels.shape)

            correct += (predicted == labels.argmax(dim = 1)).sum().item()

            loss += criterion(outputs, labels).item()
            counter += 1
    
    return loss / counter, correct / total


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, pred, mixed):
        # require pred, mixed with shape (n, c)
        return -torch.mean( torch.sum(mixed * torch.log(pred), dim = 1) )



if __name__ == "__main__":
    from torchvision.datasets import CIFAR10
    import torchvision.transforms as transforms
    import torch

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.53129727, 0.5259391, 0.52069134), (0.28938246, 0.28505746, 0.27971658))])
    trainset = CIFAR10("~/dataset/cifar10", transform = transform_train, train = True)
    loader1 = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle = True, num_workers = 2)
    loader2 = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle = True, num_workers = 2)

    for data1, data2 in zip(loader1, loader2):
        break
    m = mixup(data1, data2, alpha = 1)

    print(m[0].shape, m[1].shape)

    loss = Loss()

    print(data1[1].shape)
    onehot(data1[1])
    # loss(data1[1], m[1]).backward()
