# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 19:13:15 2016

@author: Xinghai
"""
import numpy as np
#import theano
import theano.tensor as T
#from itertools import product

import lasagne
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import InputLayer, DenseLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax
from lasagne.utils import floatX

import skimage.transform

import matplotlib.pyplot as plt


MEAN_VALUES = np.array([104, 117, 123]).reshape((3,1,1))
'''
def plot_conv_weights(layer, figsize=(6, 6)):
    """Plot the weights of a specific layer.
    Only really makes sense with convolutional layers.
    Parameters
    ----------
    layer : lasagne.layers.Layer
    """
    W = layer.W.get_value()
    shape = W.shape
    nrows = np.ceil(np.sqrt(shape[0])).astype(int)
    ncols = nrows

    for feature_map in range(shape[1]):
        figs, axes = plt.subplots(nrows, ncols, figsize=figsize)

        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')

        for i, (r, c) in enumerate(product(range(nrows), range(ncols))):
            if i >= shape[0]:
                break
            axes[r, c].imshow(W[i, feature_map], cmap='gray',
                              interpolation='nearest')
    return plt
'''
def prep_image(im):
    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
        im = np.repeat(im, 3, axis=2)
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (IMAGE_W, w*IMAGE_W/h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*IMAGE_W/w, IMAGE_W), preserve_range=True)

    # Central crop
    h, w, _ = im.shape
    im = im[h//2-IMAGE_W//2:h//2+IMAGE_W//2, w//2-IMAGE_W//2:w//2+IMAGE_W//2]
    
    rawim = np.copy(im).astype('uint8')
    
    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    
    # Convert RGB to BGR
    im = im[::-1, :, :]

    im = im - MEAN_VALUES
    return rawim, floatX(im[np.newaxis])

net = {}
IMAGE_W = 256
net['input'] = InputLayer((1, 1, IMAGE_W, IMAGE_W))
net['conv1_1'] = ConvLayer(net['input'], 64, 8, pad=1, flip_filters=False)
net['output'] = DenseLayer(net['conv1_1'],  num_units=10, nonlinearity=softmax)

photo = plt.imread('b.jpg')
rawim, photo = prep_image(photo)
plt.imshow(rawim)

layers = ['conv1_1']
layers = {k: net[k] for k in layers}

input_im_theano = T.tensor4()
outputs = lasagne.layers.get_output(net['output'], input_im_theano, deterministic= True)
#plot_conv_weights(net['conv1_1'])



