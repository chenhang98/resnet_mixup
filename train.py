# coding: utf-8
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch
import csv
import sys
import time
import os

from network import ResNet6n
from utils import test, Loss, mixup

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '4'


if __name__ == "__main__":
    start_time = time.time()

    # load data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    trainset = CIFAR10("~/dataset/cifar10", transform = transform_train, train = True)
    loader1 = torch.utils.data.DataLoader(trainset, batch_size = 128, shuffle = True, num_workers = 2)
    loader2 = torch.utils.data.DataLoader(trainset, batch_size = 128, shuffle = True, num_workers = 2)

    testset = CIFAR10("~/dataset/cifar10", transform = transform_test, train = False)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 64, shuffle = False, num_workers = 2)

    # write header
    with open('log.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "train_loss", "val_loss", "acc", "val_acc"])

    # device_ids = [2,]
    # build model and optimizer
    # model = nn.DataParallel(ResNet6n(10, n = 18), device_ids = device_ids)
    model = ResNet6n(10, n = 18)
    model.cuda()
    criterion = Loss()
    criterion.cuda()
    # model.load_state_dict(torch.load("weights.pkl"))


    # train
    i = 0
    correct, total = 0, 0
    train_loss, counter = 0, 0

    for epoch in range(1000000):
        # iteration over all train data
        for (data1, data2) in zip(loader1, loader2):
            # update lr
            if i == 0:
                optimizer = optim.SGD(model.parameters(), lr = 1e-1, weight_decay = 1e-4, momentum = 0.9)
            elif i == 32000:
                optimizer = optim.SGD(model.parameters(), lr = 1e-2, weight_decay = 1e-4, momentum = 0.9)
            elif i == 48000:
                optimizer = optim.SGD(model.parameters(), lr = 1e-3, weight_decay = 1e-4, momentum = 0.9)
            elif i == 64000:
                end_time = time.time()
                print("total time %.1f h" %((end_time - start_time)/3600))
                sys.exit(0)

            # shift to train mode
            model.train()
            
            # get the inputs
            inputs, labels = mixup(data1, data2, 0.2)
            inputs = inputs.cuda()
            labels = labels.cuda()
            
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # count acc,loss on trainset            
            total += labels.size(0)
            predicted = outputs.data.argmax(dim = 1)
            correct += (predicted == labels.argmax(dim = 1)).sum().item()        
            train_loss += loss.item()
            counter += 1

            if i % 100 == 0:
                # get acc,loss on trainset
                acc = correct / total
                train_loss /= counter
                
                # test
                val_loss, val_acc = test(model, testloader, criterion)

                print('iteration %d , epoch %d:  loss: %.4f  val_loss: %.4f  acc: %.4f  val_acc: %.4f' 
                      %(i, epoch, train_loss, val_loss, acc, val_acc))
                
                # save logs and weights
                with open('log.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([i, train_loss, val_loss, acc, val_acc])
                torch.save(model.state_dict(), 'weights.pkl')
                    
                # reset counters
                correct, total = 0, 0
                train_loss, counter = 0, 0

            i += 1
