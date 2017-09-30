from __future__ import division,print_function
import math, os, json, sys, re

# import cPickle as pickle  # Python 2
import pickle  # Python3

from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from operator import itemgetter, attrgetter, methodcaller
from collections import OrderedDict
import itertools
from itertools import chain

import pandas as pd
import PIL
from PIL import Image
from numpy.random import random, permutation, randn, normal, uniform, choice
from numpy import newaxis
import scipy
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import imread
from sklearn.metrics import confusion_matrix
import bcolz
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE

from IPython.lib.display import FileLink

import theano
from theano import shared, tensor as T
from theano.tensor.nnet import conv2d, nnet
from theano.tensor.signal import pool

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional
from keras.layers import SpatialDropout1D, Concatenate  # Keras2

from keras.layers import TimeDistributed, Activation, SimpleRNN, GRU
from keras.layers.core import Flatten, Dense, Dropout, Lambda

# from keras.regularizers import l2, activity_l2, l1, activity_l1  # Keras1
from keras.regularizers import l2, l1  # Keras2

from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam

# from keras.utils.layer_utils import layer_from_config  # Keras1
from keras.layers import deserialize  # Keras 2
from keras.layers.merge import dot, add, concatenate  # Keras2
from keras.metrics import categorical_crossentropy, categorical_accuracy
from keras.layers.convolutional import *
from keras.preprocessing import image, sequence
from keras.preprocessing.text import Tokenizer

from vgg16 import *
from vgg16bn import *
np.set_printoptions(precision=4, linewidth=100)


to_bw = np.array([0.299, 0.587, 0.114]) # has shape of (3,)

# rollaxis moves axis, but I don't understand why it is done
# in order for .dot(to_bw) to work, the last axis must be 3
def gray(img):
    if K.image_dim_ordering() == 'tf': # channels last
        return np.rollaxis(img, 0, 1).dot(to_bw) # puts 0th axis in 1st position [why?]
    else: # theano - channels first
        return np.rollaxis(img, 0, 3).dot(to_bw) # puts 0th axis in 3rd position [seems correct]

# converts to unsigned integer (0-255)
def to_plot(img):
    if K.image_dim_ordering() == 'tf':
        return np.rollaxis(img, 0, 1).astype(np.uint8)
    else:
        return np.rollaxis(img, 0, 3).astype(np.uint8)

def plot(img):
    plt.imshow(to_plot(img))

# shortcut routines
def floor(x):
    return int(math.floor(x))
def ceil(x):
    return int(math.ceil(x))


def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            # make 1st axis the last one, same as np.rollaxis(ims, 1, 4)
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

# mx is maximum
def do_clip(arr, mx):
    clipped = np.clip(arr, (1-mx)/1, mx) # why divide by 1?
    # renormalize all values, so that sum is 1
    return clipped/clipped.sum(axis=1)[:, np.newaxis] # inserts additional axis at end

# been here, done that
def get_batches(dirname, gen=image.ImageDataGenerator(), shuffle=True, batch_size=4, class_mode='categorical',
                target_size=(224,224)):
    return gen.flow_from_directory(dirname, target_size=target_size,
            class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)


def onehot(x):
    return to_categorical(x) # keras.utils.np_utils.to_categorical

# layer.__class__.__name__ is a string
# layer.get_config() is a dict
def wrap_config(layer):
    return {'class_name': layer.__class__.__name__, 'config': layer.get_config()}



'''
This creates a new layer from layer config.
as per https://keras.io/layers/about-keras-layers/
https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/contrib/keras/python/keras/metrics.py
https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/contrib/keras/python/keras/utils/generic_utils.py

Deserialization = restoring an object from a serial
representation and ensuring the invariants of the object.
Deserialization can be thought of a separate constructor for the object. 
'''
def copy_layer(layer): return deserialize(wrap_config(layer))  # Keras2


def copy_layers(layers): return [copy_layer(layer) for layer in layers]

# nice!
def copy_weights(from_layers, to_layers):
    for from_layer,to_layer in zip(from_layers, to_layers):
        to_layer.set_weights(from_layer.get_weights())


def copy_model(in_model):
    # will break is model is not Sequential
    layers_list = copy_layers(in_model.layers)
    out_model = Sequential(layers_list)
    copy_weights(in_model.layers, out_model.layers)
    return out_model

# adds new layer AT the index (starts at 0 of course)
# note that layers are passed by reference
def insert_layer(in_model, new_layer, index):
    out_model = Sequential()
    for i,layer in enumerate(in_model.layers):
        if i==index: out_model.add(new_layer)
        copied = copy_layer(layer)  # Keras2
        out_model.add(copied)
        copied.set_weights(layer.get_weights())
    return out_model

# ??? some kind of update of weights
def adjust_dropout(weights, prev_p, new_p):
    scal = (1-prev_p)/(1-new_p)
    return [o*scal for o in weights]


def get_data(path, target_size=(224,224)):
    # get_batches returns gen.flow_from_directory whichis a generator
    batches = get_batches(path, shuffle=False, batch_size=1, class_mode=None, target_size=target_size)
    # iterate through all samples in flow_from_directory generator and concat in numpy array
    return np.concatenate([batches.next() for i in range(batches.samples)])  # Keras2


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# save numpy array to disk
def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

# load numpy array from disk
def load_array(fname):
    return bcolz.open(fname)[:]

# ?!?!?!
def mk_size(img, r2c):
    r,c,_ = img.shape
    curr_r2c = r/c
    new_r, new_c = r,c
    if r2c>curr_r2c:
        new_r = floor(c*r2c)
    else:
        new_c = floor(r/r2c)
    arr = np.zeros((new_r, new_c, 3), dtype=np.float32)
    r2=(new_r-r)//2
    c2=(new_c-c)//2
    arr[floor(r2):floor(r2)+r,floor(c2):floor(c2)+c] = img
    return arr

# ?!?!?!
def mk_square(img):
    x,y,_ = img.shape
    maxs = max(img.shape[:2])
    y2=(maxs-y)//2
    x2=(maxs-x)//2
    arr = np.zeros((maxs,maxs,3), dtype=np.float32)
    arr[floor(x2):floor(x2)+x,floor(y2):floor(y2)+y] = img
    return arr

# finetune vgg apparently
def vgg_ft(out_dim):
    vgg = Vgg16()
    vgg.ft(out_dim)
    model = vgg.model
    return model

# finetune vgg with batchnorm
def vgg_ft_bn(out_dim):
    vgg = Vgg16BN()
    vgg.ft(out_dim)
    model = vgg.model
    return model


def get_classes(path):
    batches = get_batches(path+'train', shuffle=False, batch_size=1) # flow_from_directory
    val_batches = get_batches(path+'valid', shuffle=False, batch_size=1)
    test_batches = get_batches(path+'test', shuffle=False, batch_size=1)
    return (val_batches.classes, batches.classes, onehot(val_batches.classes), onehot(batches.classes),
        val_batches.filenames, batches.filenames, test_batches.filenames)

# splits at where the layer_type ends
def split_at(model, layer_type):
    layers = model.layers
    # returns index of last layer of layer_type in the model
    layer_idx = [index for index,layer in enumerate(layers) if type(layer) is layer_type][-1]
    return layers[:layer_idx+1], layers[layer_idx+1:]

# used for data augmentation, mixes original with augmented data
class MixIterator(object):
    def __init__(self, iters):
        self.iters = iters
        self.multi = type(iters) is list # iters can be also a list of iterators
        if self.multi:
            self.N = sum([it[0].N for it in self.iters])
        else:
            self.N = sum([it.N for it in self.iters])

    def reset(self):
        for it in self.iters: it.reset() # reset iterator to the beginning?

    def __iter__(self):
        return self

    def next(self, *args, **kwargs):
        if self.multi:
            nexts = [[next(it) for it in o] for o in self.iters]
            n0 = np.concatenate([n[0] for n in nexts])
            n1 = np.concatenate([n[1] for n in nexts])
            return (n0, n1)
        else:
            nexts = [next(it) for it in self.iters]
            n0 = np.concatenate([n[0] for n in nexts])
            n1 = np.concatenate([n[1] for n in nexts])
            return (n0, n1)

