# -*- coding: utf-8 -*-

import os
from os.path import join


def convert(train=True):
    root = os.getcwd()
    txt_root = join(root, "train.txt")
    if(train):
        f = open(txt_root, 'w')
        data_path = join(root, 'train')
        if(not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i in range(40):
            for j in range(10):
                img_path = join(data_path, 's' + str(i + 1), str(j + 1) + ".pgm")
                f.write(img_path+' '+str(i)+'\n')      
        f.close()


if __name__ == '__main__':
    convert()
