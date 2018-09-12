# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 08:41:10 2018

@author: pauln
"""

import numpy as np
import numpy.random
import matplotlib.pyplot as plt





#%% Generate Brownina Motion

numframes = 200
height = 128
width = 128
pad = 50

motion = np.zeros((numframes, width + pad, height + pad))

size = 10
step = 2
coord = np.array([(width+pad)//2 ,(height+pad)//2])
motion[0][coord[0]-size//2 : coord[0]+size//2 + 1,coord[1]-size//2:coord[1]+size//2 + 1] = 1

for i in range(1,numframes):
    coord[0] += int(np.round(np.random.normal(0,step))) 
    coord[1] += int(np.round(np.random.normal(0,step)))        
    motion[i,:,:] *= 0
    motion[i][coord[0]-size//2 : coord[0]+size//2 + 1,coord[1]-size//2:coord[1]+size//2 + 1] = 1
        
motion = motion[:,(width+pad)//2 - width//2:(width+pad)//2 + width - width//2, (height+pad)//2 - height//2:(height+pad)//2 + height - height//2]
movArray = motion
plt.imshow(motion[-1])
#%% John Conway's Game of Life
# "pentomino" initial configuration

numframes = 200
size = 4
height = 128//size
width = 128//size
pad = 0

motion = np.zeros((numframes, width + pad, height + pad))

motion[0][width//2-1:width//2+2,height//2] = 1
motion[0][width//2,height//2 - 1] = 1
motion[0][width//2 - 1,height//2 + 1] = 1

for i in range(1,numframes):
    for x in range(width + pad):
        for y in range(height + pad):
            neighb = int(np.sum(motion[i-1][x-1:x+2,y-1:y+2]))
            if ((motion[i-1][x,y] == 0) and (neighb == 3)):
                motion[i][x,y] = 1
            elif ((motion[i-1][x,y] == 1) and (neighb not in [3,4])):
                motion[i][x,y] = 0
            else:
                motion[i][x,y] = motion[i-1][x,y]


motion = motion[:,(width+pad)//2 - width//2:(width+pad)//2 + width - width//2, (height+pad)//2 - height//2:(height+pad)//2 + height - height//2]

movArray = np.zeros((numframes, 128, 128))
for i in range(motion.shape[0]):
    movArray[i] = np.kron(motion[i], np.ones((size,size)))

motion = movArray

#%% John Conway's Game of Life
# "die hard" initial configuration

numframes = 200
size = 4
height = 128//size
width = 128//size
pad = 0

motion = np.zeros((numframes, width + pad, height + pad))

motion[0][width//2:width//2+2,height//2] = 1
motion[0][width//2 + 1,height//2 + 4:height//2 + 7] = 1
motion[0][width//2,height//2 - 1] = 1
motion[0][width//2 - 1,height//2 + 5] = 1

for i in range(1,numframes):
    for x in range(width + pad):
        for y in range(height + pad):
            neighb = int(np.sum(motion[i-1][x-1:x+2,y-1:y+2]))
            if ((motion[i-1][x,y] == 0) and (neighb == 3)):
                motion[i][x,y] = 1
            elif ((motion[i-1][x,y] == 1) and (neighb not in [3,4])):
                motion[i][x,y] = 0
            else:
                motion[i][x,y] = motion[i-1][x,y]


motion = motion[:,(width+pad)//2 - width//2:(width+pad)//2 + width - width//2, (height+pad)//2 - height//2:(height+pad)//2 + height - height//2]

movArray = np.zeros((numframes, 128, 128))
for i in range(motion.shape[0]):
    movArray[i] = np.kron(motion[i], np.ones((size,size)))

motion = movArray

#%% Circular trajectory
numframes = 200
size = 4
height = 128
width = 128
pad = 0




motion = np.zeros((numframes, width + pad, height + pad))

size = 10
edge = 10
radius = height//2 - edge
# Number of circles in the training video
n_circle = 2
# Standard deviation of the noise on the coordinates
noise = 0
peri = numframes//n_circle
coord = np.array([(width+pad)//2 ,(height+pad)//2])
motion[0][coord[0]-size//2 : coord[0]+size//2,coord[1]-size//2:coord[1]+size//2] = 1

for i in range(1,numframes):
     
    x = width//2 + np.round(radius*np.cos(2*np.pi*i/peri))
    y = height//2 + np.round(radius*np.sin(2*np.pi*i/peri))
    
    coord[0] = x + np.round(np.random.normal(0,noise))
    coord[1] = y + np.round(np.random.normal(0,noise))

    motion[i,:,:] *= 0
    motion[i][coord[0]-size//2 : coord[0]+size//2 + 1,coord[1]-size//2:coord[1]+size//2 + 1] = 1


        
motion = motion[:,(width+pad)//2 - width//2:(width+pad)//2 + width - width//2, (height+pad)//2 - height//2:(height+pad)//2 + height - height//2]
movArray = np.zeros((numframes, 128, 128))
movArray = motion

