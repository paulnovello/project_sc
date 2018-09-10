from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam


import matplotlib.pyplot as plt

import sys

import numpy as np
import pandas as pd

#%% Train the GAN. 
# During the training, images will be generated each 1000 steps in /images directory

class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 40
        self.img_cols = 40
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        # Change the latent dimension here
        self.latent_dim = 2

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
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

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

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
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):
        

        data = pd.read_csv('SquaresBnW.txt', header = None, sep=' ')
        data = data.as_matrix()

        # Load the dataset
        X_train = data[:40000,2:].astype('float32')

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
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
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
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=40000, batch_size=32, save_interval=1000)
    
#%% generate 10 images

N = 10
pred = np.zeros((N,40,40))
latent = np.zeros((N,1,dcgan.latent_dim))

for i in range(N):
    latent[i] = np.random.normal(0,1,(1,dcgan.latent_dim))
    pred[i] = np.reshape(dcgan.generator.predict(latent[i]),(40,40))

#%% plot them  
plt.figure(figsize = (10,10))
for i in range(N):
    plt.subplot(3,3,i+1)
    plt.imshow(pred[i], cmap = "gray")
    
plt.show()   
    



#%% generate N images

N = 10000
pred = np.zeros((N,40,40))
latent = np.zeros((N,1,dcgan.latent_dim))

for i in range(N):
    latent[i] = np.random.normal(0,1,(1,dcgan.latent_dim))
    pred[i] = np.reshape(dcgan.generator.predict(latent[i]),(40,40))

label = np.zeros((N,1))
    
# since the squares have the same size, and the shape is well reproduced, 
# we can have an estimation of the number of squares in each image
for i in range(N):
    label[i] = (np.sum(0.5*(pred[i]+1))+25/2)//25
    
    



#%% for a 2-D latent space
colormap = np.array(['r','g','b','c','m','y','k','tab:orange','tab:pink','tab:gray','tab:purple'])
latent = np.reshape(latent,(N,dcgan.latent_dim))
label = np.array(label, dtype='int8')
label = np.reshape(label,N)


x = latent[:,0]
y = latent[:,1]

plt.scatter(x,y, c=colormap[label])
plt.show()

#%% for a 100-D latent space 
# norm increase

plt.figure(figsize = (10,10))
for i in range(N):
    plt.subplot(3,3,i+1)
    plt.imshow(np.reshape(dcgan.generator.predict((1+0.1*i)*latent[0]),(40,40)), cmap = "gray")
    
plt.show()  



#%% for a 100-D latent space 
# linear combinations

# since the linear combinations don't work with all generated images, we tried manually with different 
# vectors in the array "latent". Then, to find suitable linear combination coefficients, we plotted a
# a grid of 10*10 images, such that an image located at (i,j) is generated from the vector
# 0.1*i*vec1 + 0.1*j*vec2.
# Thus, we can chose visually the coefficients of the linear combination.
vec1 = np.reshape(latent[np.where(label == 1)[0]][0],(1,dcgan.latent_dim))
vec2 = np.reshape(latent[np.where(label == 1)[0]][1],(1,dcgan.latent_dim))

N = 100


plt.figure(figsize = (15,15))
for i in range(N):
    size = 10
    plt.subplot(size,size,i+1)
    plt.imshow(np.reshape(dcgan.generator.predict((i%size +1)*0.1*vec1 + (i//size + 1)*0.1*vec2),(40,40)), cmap = "gray")
plt.show()  