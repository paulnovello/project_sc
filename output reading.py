# -*- coding: utf-8 -*-
"""
Created on Tue May 22 10:06:37 2018

@author: pauln
"""

import pandas as pd
import numpy as np
data = pd.read_csv('encoded_imgs.txt', header = None, sep=' ')
data = data.as_matrix()
enc = pd.read_csv('encoded.txt', header = None, sep=' ')
enc = enc.as_matrix()
data2 = pd.read_csv('simple_squares.txt', header = None, sep=' ')
data2 = data2.as_matrix()
x_test = data2[40000:50000]

res = 40

import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(res, res))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(data[i].reshape(res, res))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

enc[0][np.where(enc[0] ==0)].shape