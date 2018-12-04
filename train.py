# -*- coding: utf-8 -*-

import os
from os.path import join
import torch
from torch.autograd import Variable
import random
import torchvision
import linecache
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import PIL.ImageOps
import matplotlib.pyplot as plt    
from Dataset import *
from Model.SiameseNet import *
from Model.ContrastiveLoss import *
import torch.optim as optim


class Config():
    root = os.getcwd()
    txt_root = join(root, "train.txt")
    train_batch_size = 32
    train_number_epochs = 30


# Helper functions
def imshow(img, text=None, should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show() 


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()

# Visualising some of the data
"""
train_data = MyDataset(txt = Config.txt_root,transform=transforms.Compose(
            [transforms.Resize((100,100)),transforms.ToTensor()]), should_invert=False)

train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True)
it = iter(train_loader)
p1, p2, label = it.next()
example_batch = it.next()
concatenated = torch.cat((example_batch[0],example_batch[1]),0)
imshow(torchvision.utils.make_grid(concatenated))
print(example_batch[2].numpy())
"""

if __name__ == '__main__':
    train_data = MyDataset(txt = Config.txt_root,transform=transforms.Compose(
            [transforms.Resize((100,100)),transforms.ToTensor()]), should_invert=False)
    train_dataloader = DataLoader(dataset=train_data, shuffle=True, num_workers=2, batch_size = Config.train_batch_size)

    net = SiameseNetwork().cuda()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)


    counter = []
    loss_history =[]
    iteration_number =0

    for epoch in range(0, Config.train_number_epochs):
        for i, data in enumerate(train_dataloader, 0):
            img0, img1, label = data
            img0, img1, label = Variable(img0).cuda(), Variable(img1).cuda(), Variable(label).cuda()
            output1, output2 = net(img0, img1)
            optimizer.zero_grad()
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            
            if i%10 == 0:
                  print("Epoch:{},  Current loss {}\n".format(epoch,loss_contrastive.data[0]))
                  iteration_number += 10
                  counter.append(iteration_number)
                  loss_history.append(loss_contrastive.data[0])
    show_plot(counter, loss_history)


