# -*- coding: utf-8 -*-
"""
Created on Wed May 16 09:36:26 2018

@author: pauln
"""

import numpy as np
import numpy.random
import matplotlib.pyplot as plt
#%%

N = 50000
res = 40
# to add for Poisson background
#base = np.random.poisson(5,(N,res,res))
baseref = np.zeros((N,res,res)) 
labelc = np.zeros(N)
labels = np.zeros(N)
rep = [0, 1, 0, 1, 0, 1, 0, 1, 0]

for i in range(baseref.shape[0]):
    nsq = np.random.randint(1,10)
    size = 5
    count = 0
    while np.sum(baseref[i]) == 0:
        j = 0
        for k in range(nsq):
            x = np.random.randint(0,res-1)
            y = np.random.randint(0,res-1)
            if ((x + size < res) and (y + size < res)):
                if (np.max(baseref[i][max(0,x-1):x + size + 1,max(y-1,0):y + size + 1]) == 0):
                    # to add for Poisson background
                    # base[i][x:x+size,y:y+size] = rep[j]*np.random.poisson(1) + (1-rep[j])*np.random.poisson(9)
                    baseref[i][x:x+size,y:y+size] += 1
                    count += 1
                    j+= 1
    labelc[i] = count
    labels[i] = size

                                                            


base = baseref    
base = np.reshape(base,(N,res*res))  
base = np.c_[labelc,labels,base]
np.savetxt("SquaresBnW.txt",base, fmt = "%i")
#np.savetxt("SquaresPoisson.txt",base, fmt = "%i")

#%%

res = 40
x_train = base[:40000,2:].astype('float32')
x_train = np.reshape(x_train, (len(x_train), res, res, 1))

n = 25  # how many digits we will display
plt.figure(figsize=(15, 15))
for i in range(n):
    # display original
    ax = plt.subplot(5, 5, i + 1)
    plt.imshow(x_train[i].reshape(res, res))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
