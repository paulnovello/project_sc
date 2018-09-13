### Applications of Neural Networks for Unsupervised Representation Learning

The scripts are partitioned into boxes that have to be executed one by one. 
Thus it is easier to inspect the different intermediate objects of the computations.

## Application 1

First of all, the user will have to generate a data file with generate_squares.py, because the size of the obtained
dataset is too large to be uploaded on github.

# Autoencoders

In Autoencoders.py, there are two sets of boxes, one for training and applying the autoencoder, and the other for the classifier. 
Each box is labelled in order to know in what case it has to be executed.

# GANs

In dcgan.py, generated images will be created in the directory /images throughout the training.
Then, manipulations can be performed with the code contained in the following boxes.
The code is greatly inspired from https://github.com/eriklindernoren/Keras-GAN/tree/master/dcgan

# SDGANs

In sdgan.py, you will find a very similar code to dcgan.py, except that there is no manipulation boxes since the results
of the training are not satisfying.

## Application 2

For this application, there is a single training script, and the other are objects created in https://github.com/coxlab/prednet to 
implement the network. First, you have to create the video you want to train the network on with generate_video.py, and then
you can train and manipulate the network using train.py, on the same console.


