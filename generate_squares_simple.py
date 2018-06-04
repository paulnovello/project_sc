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
#base = np.random.poisson(5,(N,res,res))
baseref = np.zeros((N,res,res)) 
label = np.zeros(N)

for i in range(baseref.shape[0]):
    nsq = np.random.randint(1,10)
    shape = np.random.randint(1,10)
    count = 0
    while np.sum(baseref[i]) == 0:
        for k in range(nsq):
            x = np.random.randint(0,res-1)
            y = np.random.randint(0,res-1)
            if ((x + shape < res) and (y + shape < res)):
                if (np.max(baseref[i][max(0,x-1):x + shape + 1,max(y-1,0):y + shape + 1]) == 0):
#                    decision = np.random.randint(2)
#                    base[i][x:x+shape,y:y+shape] = decision*np.random.poisson(1) + (1-decision)*np.random.poisson(9)
                    baseref[i][x:x+shape,y:y+shape] += 1
                    count += 1
    label[i] = count

                                                   

                    
     #+ decision*np.random.poisson(2) + (1-decision)*np.random.poisson(5)               


base = baseref    
base = np.reshape(base,(N,res*res))  
base = -1*(base - 1)
base = np.c_[label,base]
np.savetxt("simple_squaresInv.txt",base, fmt = "%i")

#%%

res = 40
x_train = base[:40000,].astype('float32')
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
    
    #%%
 c = 0   
for i in range(base.shape[0]):
    if len(np.where(base[i] == 3)[0]) != 0:
        test = i
        c+=1
        
plt.imshow(np.reshape(base[test],(28,28)))