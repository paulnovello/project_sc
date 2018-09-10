# -*- coding: utf-8 -*-
"""
Created on Wed May 16 15:27:55 2018

@author: pauln
"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, ZeroPadding2D
from keras.models import Model
from keras import layers, models, optimizers
from keras import backend as K
from keras import regularizers
import keras
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture




#%% Autoencoder model

# height/width of the images
res = 40
# dimension of the latent space
encoding_dim = 2000  

input_img = Input(shape=(res, res, 1))  

x = Conv2D(9, (9, 9), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(9, (9, 9), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(20, (5, 5), activation='relu', padding='same')(x)
x = Flatten()(x)
encoded = Dense(encoding_dim, activation='relu')(x)

x = Dense(1600, activation='relu')(encoded)
x = Reshape((40,40,1))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding = "same")(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# to build the autoencoder model
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# to build the encoder model
encoder = Model(input_img, encoded)

# to build the decoder model
input_dec = Input(shape=(encoding_dim,))
x = autoencoder.layers[-5](input_dec)
x = autoencoder.layers[-4](x)
x = autoencoder.layers[-3](x)
x = autoencoder.layers[-2](x)
dec_output = autoencoder.layers[-1](x)
decoder = Model(input_dec, dec_output)

#%% Classifier model with the same structure than the previous encoder

# Use this code if 

# this is the height/width of the images
res = 40
# number of categories to classify
n_cat = 10  

input_img = Input(shape=(res, res, 1))  

x = Conv2D(9, (9, 9), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(9, (9, 9), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(20, (5, 5), activation='relu', padding='same')(x)
x = Flatten()(x)
encoded = Dense(2000, activation='relu')(x)
category = Dense(n_cat, activation='softmax')(encoded)

# to build the classifier model
classifier = Model(input_img, category)
classifier.compile(optimizer='adadelta', loss='categorical_crossentropy')

# to build the encoder model
encoder = Model(input_img, encoded)


#%% import the data

import pandas as pd
import numpy as np
data = pd.read_csv('SquaresPoisson.txt', header = None, sep=' ')
data = data.as_matrix()

x_train = data[:40000,2:].astype('float32')
x_test = data[40000:50000,2:].astype('float32')
label_test = data[40000:50000,0]
label_train = data[:40000,0]
size_test = data[40000:50000,1]
size_train = data[:40000,1]
x_train = np.reshape(x_train, (len(x_train), res, res, 1))/np.max(x_train)
x_test = np.reshape(x_test, (len(x_test), res, res, 1))/np.max(x_test)

#%% Training for autoencoder


autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=100,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)

#%% Training for classifier

y_train = keras.utils.to_categorical(label_train)
y_test = keras.utils.to_categorical(label_test )

classifier.fit(x_train, y_train,
               epochs=50,
               batch_size=100,
               shuffle=True,
               validation_data=(x_test, y_test))

encoded_imgs = encoder.predict(x_test)


#%% Display reconstruction for the Autoencoder

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
    plt.imshow(decoded_imgs[i].reshape(res, res))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


#%% PCA on encoded data

from scipy.sparse.linalg import eigs

def myPCA(M,X):
    m = np.mean(X, axis = 0)
    Xp = np.array(X)   
    Xp = Xp - m
    W, V = eigs(1/len(Xp.T)*Xp.T.dot(Xp),M)
    return(W,np.real(V))

eigen, direc = myPCA(148,encoded_imgs) 
eigen = np.real(eigen)

#%% histogram of eigenvalues
plt.figure(figsize=(20, 20))
plt.hist(eigen, bins = 150, range = (0,np.max(eigen)))
plt.show()

neff = eigen[np.where(eigen >= 0.015)].shape[0]



#%% projection + graph

colormap = np.array(['r','g','b','c','m','y','k','tab:orange','tab:pink','tab:gray'])

dirproj = direc/eigen
x1 = dirproj[:,0]
x2 = dirproj[:,1]
x = np.sum(encoded_imgs*x1, axis = 1)
y = np.sum(encoded_imgs*x2, axis = 1)
ymin = np.min(y)
ymax = np.max(y)
xmin = np.min(x)
xmax = np.max(x)

# colour plot of the projection of the dataset on PC1 and PC2
plt.figure(figsize=(10, 10))
plt.scatter(x,y, c=colormap[label_test])
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.show()

# plot of the projection of the dataset on PC1 and PC2 for each different number of squares
for i in np.unique(label_test):
    plt.scatter(x[np.where(label_test == i)],y[np.where(label_test == i)], c=colormap[i])
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.show()

