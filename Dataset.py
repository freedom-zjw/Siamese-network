# -*- coding: utf-8 -*-

import torch
import random
import linecache
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import PIL.ImageOps   


# ready the dataset, Not use ImageFolder as the author did
class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, should_invert=False):
        self.transform = transform
        self.target_transform = target_transform
        self.should_invert = should_invert
        self.txt = txt

    def __getitem__(self, index):
        line = linecache.getline(self.txt, random.randint(1, self.__len__()))
        line.strip('\n')
        img0_list = line.split()
        should_get_same_class = random.randint(0, 1) 
        if should_get_same_class:
            while True:
                img1_list = linecache.getline(
                    self.txt, random.randint(1, self.__len__())
                ).strip('\n').split()
                if img0_list[1] == img1_list[1]:
                    break
        else:
            img1_list = linecache.getline(
                self.txt, random.randint(1, self.__len__())
            ).strip('\n').split()
            
        img0 = Image.open(img0_list[0])
        img1 = Image.open(img1_list[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1, torch.from_numpy(np.array([int(img1_list[1]!=img0_list[1])],dtype=np.float32))
    
    def __len__(self):
        fh = open(self.txt, 'r')
        num = len(fh.readlines())
        fh.close()
        return num


if __name__ == '__main__':
    pass
