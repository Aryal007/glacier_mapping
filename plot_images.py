#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 11:29:38 2020

@author: mibook
"""

import matplotlib.pyplot as plt
# import numpy as np
import glob
import matplotlib.image as mpimg

path = "/home/mibook/Desktop/Fall 2020/glacier mapping/images/pred_images/mlp/preds/"

files = sorted(glob.glob(path+'*'))
images = []
for img_path in files:
    images.append(mpimg.imread(img_path))

fig = plt.figure(figsize=(9,14))

columns = 10
rows = 11
for i in range(1, len(images)+1):
    img = images[i-1]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
    plt.axis('off')
fig.tight_layout()
fig.subplots_adjust(wspace=0.1)

plt.show()