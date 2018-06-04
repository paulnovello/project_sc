# -*- coding: utf-8 -*-
"""
Created on Wed May 16 15:27:55 2018

@author: pauln
"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, ZeroPadding2D
from keras.models import Model
from keras import backend as K
from keras import regularizers
from keras import optimizers
import tensorflow as tf


#%% Convolutional

res = 40
# this is the size of our encoded representations
encoding_dim = 2  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

input_img = Input(shape=(res, res, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(9, (9, 9), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(9, (9, 9), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(2, (5, 5), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Flatten()(x)
encoded = Dense(256, activation='relu')(x)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(x)
encoded = Dense(encoding_dim, activation='relu')(encoded)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(1024, activation='relu')(decoded)
decoded = Dense(res*res, activation='sigmoid')(decoded)
decoded = Reshape((res,res,1))(decoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#autoencoder.compile(optimizer='SGD', loss='mse')

encoder = Model(input_img, encoded)

input_dec = Input(shape=(encoding_dim,))
x = autoencoder.layers[-6](input_dec)
x = autoencoder.layers[-5](x)
x = autoencoder.layers[-4](x)
x = autoencoder.layers[-3](x)
x = autoencoder.layers[-2](x)
dec_output = autoencoder.layers[-1](x)
decoder = Model(input_dec, dec_output)

#%% full convolutional

res = 40
# this is the size of our encoded representations
encoding_dim = 2000  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

input_img = Input(shape=(res, res, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(9, (9, 9), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(9, (9, 9), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(20, (5, 5), activation='relu', padding='same')(x)
x = Flatten()(x)
encoded = Dense(encoding_dim, activation='relu')(x)
#encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

#x = Conv2D(2, (3, 3), activation='relu', padding='same')(encoded)
#x = UpSampling2D((2, 2))(x)
x = Dense(100, activation='relu')(encoded)
x = Reshape((10,10,1))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding = "same")(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#autoencoder.compile(optimizer='SGD', loss='mse')

encoder = Model(input_img, encoded)



input_dec = Input(shape=(encoding_dim,))
x = autoencoder.layers[-7](input_dec)
x = autoencoder.layers[-6](x)
x = autoencoder.layers[-5](x)
x = autoencoder.layers[-4](x)
x = autoencoder.layers[-3](x)
x = autoencoder.layers[-2](x)
dec_output = autoencoder.layers[-1](x)
decoder = Model(input_dec, dec_output)


#%% Sparse

res = 40
# this is the size of our encoded representations
encoding_dim = 50  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

input_img = Input(shape=(res, res, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(9, (5, 5), activation='relu', padding='same')(input_img)
#x = MaxPooling2D((2, 2), padding='same')(x)
#x = Conv2D(9, (3, 3), activation='relu', padding='same')(x)
#x = MaxPooling2D((2, 2), padding='same')(x)
#x = Conv2D(9, (3, 3), activation='relu', padding='same')(x)
#x = MaxPooling2D((2, 2), padding='same')(x)
x = Flatten()(x)
#encoded = Dense(2000, activation='relu')(x)
#encoded = Dense(2000, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu',
                activity_regularizer=regularizers.l1(10e-4))(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

#decoded = Dense(2000, activation='relu')(encoded)
decoded = Dense(res*res, activation='sigmoid')(encoded)
decoded = Reshape((res,res,1))(decoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#autoencoder.compile(optimizer='SGD', loss='mse')

encoder = Model(input_img, encoded)

input_dec = Input(shape=(encoding_dim,))
x = autoencoder.layers[-2](input_dec)
dec_output = autoencoder.layers[-1](x)
decoder = Model(input_dec, dec_output)

#%% Simple Squares

import pandas as pd
import numpy as np
data = pd.read_csv('simple_squares.txt', header = None, sep=' ')
data = data.as_matrix()

x_train = data[:40000,1:].astype('float32')
x_test = data[40000:50000,1:].astype('float32')
label_test = data[40000:50000,0]
x_train = np.reshape(x_train, (len(x_train), res, res, 1))/np.max(x_train)
x_test = np.reshape(x_test, (len(x_test), res, res, 1))/np.max(x_test)

#%%


autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=100,
                shuffle=True,
                validation_data=(x_test, x_test))

#%%

encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)
decoded_2 = decoder.predict(encoder.predict(x_test))

#%%

# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

dec = 0
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(dec,dec+n):
    # display original
    ax = plt.subplot(2, n, i - dec + 1)
    plt.imshow(x_test[i].reshape(res, res))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i -dec + 1 + n)
    plt.imshow(decoded_2[i].reshape(res, res))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

#%%

max = np.max(encoded_imgs)
min = np.min(encoded_imgs)
sd = np.sqrt(np.var(encoded_imgs))
mean = np.mean(encoded_imgs)
med = np.median(encoded_imgs)

print("max = " + str(max) + "\n min = " + str(min) + "\n sd = " + str(sd)
      + "\n mean = " + str(mean) + "\n med = " + str(med)  )

#%%

n = 10
size = 30
plt.figure(figsize=(20, 20))
for i in range(n):
    for j in range(n):
        ax = plt.subplot(n, n, i*n + j + 1)
        plt.imshow(decoder.predict(np.array([[i,j]])).reshape(res, res))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
plt.show()

#%% PCA on encoded 

from scipy.sparse.linalg import eigs

def myPCA(M,X):
    m = np.mean(X, axis = 0)
    Xp = np.array(X)
    
    Xp = Xp - m
    W, V = eigs(1/len(Xp.T)*Xp.T.dot(Xp),M)
    return(W,np.real(V))

eigen, direc = myPCA(154,encoded_imgs) 
eigen = np.real(eigen)

#%% histo
plt.figure(figsize=(20, 20))
plt.hist(eigen, bins = 154, range = (0,np.max(eigen)))
plt.show()

neff = eigen[np.where(eigen >= 0.4)].shape[0]

#%% plot component

meig = direc[:,0]
madir = decoder.predict(np.array([meig.reshape(encoding_dim)]))
plt.imshow(madir.reshape((res,res)))

#%% projection + graph

dirproj = direc/eigen
x1 = dirproj[:,0]
x2 = dirproj[:,1]
x = np.sum(encoded_imgs*x1, axis = 1)
y = np.sum(encoded_imgs*x2, axis = 1)
plt.figure(figsize=(10, 10))
plt.plot(y,x, 'o')
