'''
Train PredNet on KITTI sequences. (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
'''

import os
import numpy as np
np.random.seed(123)
from six.moves import cPickle

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from prednet import PredNet
from data_utils import SequenceGenerator

#%%
numframes = 10000
comp = 250
img = Image.open("Utr-GFP dataset 3.tif")
height = img.size[0]
width = img.size[1]
imgArray = np.zeros( ( numframes, height, width  ) )
frame = 0
try:
    while 1:
        img.seek( frame )
        imgArray[frame,:,:] = img
        frame = frame + 1
except EOFError:
    img.seek( 0 )
    pass

movArray = np.zeros((250,height,width))
for i in range(250):
    movArray[i,:,:] = imgArray[numframes//250*i,:,:]
#%%

dim = movArray.shape[1]

# Data files
train_file = np.reshape(movArray[:120], (120, dim, dim, 1))
val_file = np.reshape(movArray[120:150], (30, dim, dim, 1))


# Training parameters
nb_epoch = 100
batch_size = 8
samples_per_epoch = 20
N_seq_val = 100  # number of sequences to use for validation

# Model parameters
n_channels, im_height, im_width = (1, dim, dim)
input_shape = (n_channels, im_height, im_width) if K.image_data_format() == 'channels_first' else (im_height, im_width, n_channels)
stack_sizes = (n_channels, 48, 96, 192)
R_stack_sizes = stack_sizes
A_filt_sizes = (3, 3, 1)
Ahat_filt_sizes = (3, 3, 3, 1)
R_filt_sizes = (3, 3, 3, 1)
layer_loss_weights = np.array([1., 0., 0., 0.])  # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
nt = 8  # number of timesteps used for sequences in training
time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))  # equally weight all timesteps except the first
time_loss_weights[0] = 0


prednet = PredNet(stack_sizes, R_stack_sizes,
                  A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                  output_mode='error', return_sequences=True)

inputs = Input(shape=(nt,) + input_shape) 
errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers)
errors_by_time = TimeDistributed(Dense(1, trainable=False), weights=[layer_loss_weights, np.zeros(1)], trainable=False)(errors)  # calculate weighted error by layer
errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)
final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time
model = Model(inputs=inputs, outputs=final_errors)
model.compile(loss='mean_absolute_error', optimizer='adam')

train_generator = SequenceGenerator(train_file, nt, batch_size=batch_size, shuffle=True)
val_generator = SequenceGenerator(val_file, nt, batch_size=batch_size, N_seq=N_seq_val)

lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
callbacks = [LearningRateScheduler(lr_schedule)]


history = model.fit_generator(train_generator, samples_per_epoch / batch_size, nb_epoch, callbacks=callbacks,
                validation_data=val_generator, validation_steps=N_seq_val / batch_size)

#%%
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

train_model = model
layer_config = train_model.layers[1].get_config()
layer_config['output_mode'] = 'prediction'
data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
input_shape = list(train_model.layers[0].batch_input_shape[1:])
input_shape[0] = nt
inputs = Input(shape=tuple(input_shape))
predictions = test_prednet(inputs)
test_model = Model(inputs=inputs, outputs=predictions)

test_file = np.reshape(movArray[150:], (50, dim, dim, 1))
test_generator = SequenceGenerator(test_file, nt, batch_size=batch_size, N_seq=N_seq_val)
X_test = test_generator.create_all()
X_hat = test_model.predict(X_test, batch_size)

#%%
aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
plt.figure(figsize = (20, 20*aspect_ratio))
gs = gridspec.GridSpec(2, nt)
gs.update(wspace=0., hspace=0.)


n_plot = 40
plot_idx = np.random.permutation(X_test.shape[0])[:n_plot]
for k in range(5):
    i = plot_idx[k]
    for t in range(nt):
        plt.subplot(gs[t])
        plt.imshow(np.reshape(X_test[i,t],(X_test.shape[2],X_test.shape[2])), interpolation='none')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Actual', fontsize=10)
    
        plt.subplot(gs[t + nt])
        plt.imshow(np.reshape(X_hat[i,t],(X_test.shape[2],X_test.shape[2])), interpolation='none')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Predicted', fontsize=10)
            
    plt.show()
    
#%%

plt.figure(figsize=(10, 10))
for t in range(nt):
    plt.subplot(gs[t])
    plt.imshow(movArray[t], interpolation='none')
    plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
    
plt.show()
#%%

mse_model = np.mean( (X_test[:, 1:] - X_hat[:, 1:])**2 )  # look at all timesteps except the first
mse_prev = np.mean( (X_test[:, :-1] - X_test[:, 1:])**2 )
(mse_prev - mse_model)/mse_prev


#%%

plt.plot(x,y, linestyle='-', marker='o')
plt.ylabel("accuracy")
plt.xlabel("sigma")
plt.ylim([0,1])