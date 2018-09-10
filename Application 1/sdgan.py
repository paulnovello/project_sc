# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 14:36:08 2018

@author: pauln
"""

from __future__ import print_function, division, absolute_import
import numpy as np


from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda, concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adadelta, Adam

import tensorflow as tf


import matplotlib.pyplot as plt

import sys

import numpy as np
import pandas as pd

#%% Train the GAN. 
# During the training, images will be generated each 1000 steps in /imagesSD directory

class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 40
        self.img_cols = 40
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.id_dim = 200
        self.obs_dim = 1000
        self.latent_dim = self.id_dim + self.obs_dim

        optimizer = Adam(0.0002, 0.5)


        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(2,self.latent_dim,))      
        img = self.generator(z)
        
        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 10 * 10, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((10, 10, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(2,self.latent_dim,)) 
        def slice_1(x):
            return x[:,0,:]     
        def slice_2(x):
            return x[:,1,:] 
        noise_1 = Lambda(slice_1)(noise)
        noise_2 = Lambda(slice_2)(noise)  
        img_1 = model(noise_1)
        img_2 = model(noise_2)
        img = concatenate([img_1,img_2],axis = 1)
       
        
        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())

        img = Input(shape=(2*self.img_rows, self.img_cols, self.channels))   
        def slice_1(x):
            return x[:,:self.img_cols,:,:]     
        def slice_2(x):
            return x[:,self.img_cols:,:,:] 
        img_1 = Lambda(slice_1)(img)
        img_2 = Lambda(slice_2)(img)  
        int_1 = model(img_1)
        int_2 = model(img_2)        
        x = concatenate([int_1,int_2],axis = 1)
        
        validity = Dense(1, activation='sigmoid')(x)

        return Model(img, validity)


    def train(self, epochs, batch_size=128, save_interval=50):
        

        data = pd.read_csv('simple_squaresSS.txt', header = None, sep=' ')
        data = data.as_matrix()

        # Load the dataset
        X_train = data[:40000,2:].astype('float32')
        label_train = data[:40000,0].astype('float32')

        # Rescale -1 to 1
        X_train = 2*np.reshape(X_train, (len(X_train), 40, 40, 1))/np.max(X_train) - 1

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = np.zeros((batch_size, 2*self.img_rows, self.img_cols, self.channels))
            imgs_1 = X_train[idx]
            labs = label_train[idx]
            for i in range(batch_size):
                n = np.where(label_train == labs[i])[0].shape[0]
                ind = np.random.randint(0, n)
                ind = np.where(label_train == labs[i])[0][ind]
#### ERROR HERE ?(CONCATENATED WHERE IT SHOULD BE MIXED)
                imgs[i] = np.concatenate([imgs_1[i],X_train[ind]],axis=0)
  

            # Sample noise and generate a batch of new images
            
            noise = np.zeros((batch_size, 2, self.latent_dim))
            for i in range(batch_size):
                noise_id = np.random.normal(0,1, self.id_dim)
                noise_1 = np.append(noise_id, np.random.normal(0,1,self.obs_dim))
                noise_2 = np.append(noise_id, np.random.normal(0,1,self.obs_dim))
                noise[i,0,:] = noise_1
                noise[i,1,:] = noise_2
                
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            
            noise = np.zeros((batch_size, 2, self.latent_dim))
            for i in range(batch_size):
                noise_id = np.random.normal(0,1, self.id_dim)
                noise_1 = np.append(noise_id, np.random.normal(0,1,self.obs_dim))
                noise_2 = np.append(noise_id, np.random.normal(0,1,self.obs_dim))
                noise[i,0,:] = noise_1
                noise[i,1,:] = noise_2

            gen_imgs = self.generator.predict(noise)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.zeros((r*c, 2, self.latent_dim))
        for i in range(r*c):
            noise_id = np.random.normal(0,1, self.id_dim)
            noise_1 = np.append(noise_id, np.random.normal(0,1,self.obs_dim))
            noise_2 = np.append(noise_id, np.random.normal(0,1,self.obs_dim))
            noise[i,0,:] = noise_1
            noise[i,1,:] = noise_2
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("imagesSD/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=20000, batch_size=32, save_interval=1000)