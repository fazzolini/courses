from __future__ import division, print_function

import os, json
from glob import glob
import numpy as np
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D  # Conv2D: Keras2
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image


vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1))

def vgg_preprocess(x):
    x = x - vgg_mean
    # x[:, ::-1] is equivalent to x[:, ::-1, :, :]
    # it reverses order of second dimension (channels in this case)
    return x[:, ::-1] # reverse axis rgb->bgr


class Vgg16():
    """The VGG 16 Imagenet model"""


    def __init__(self):
        self.FILE_PATH = 'http://files.fast.ai/models/'
        self.create()
        self.get_classes()


    def get_classes(self):
        fname = 'imagenet_class_index.json'
        fpath = get_file(fname, self.FILE_PATH+fname, cache_subdir='models')
        with open(fpath) as f:
            class_dict = json.load(f)
        self.classes = [class_dict[str(i)][1] for i in range(len(class_dict))]

    def predict(self, imgs, details=False):
        all_preds = self.model.predict(imgs)
        idxs = np.argmax(all_preds, axis=1)
        preds = [all_preds[i, idxs[i]] for i in range(len(idxs))]
        classes = [self.classes[idx] for idx in idxs]
        return np.array(preds), idxs, classes


    def ConvBlock(self, layers, filters):
        model = self.model
        for i in range(layers):
            model.add(ZeroPadding2D((1, 1))) # adds padding to the input volume, to keep spatial dimension same across layers
            model.add(Conv2D(filters, kernel_size=(3, 3), activation='relu'))  # Keras2
        model.add(MaxPooling2D((2, 2), strides=(2, 2))) # reduces spacial dimensions by 2


    def FCBlock(self):
        model = self.model
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))



    def create(self):
        model = self.model = Sequential() # why double assignment? just `model` only for scope of this method (to make shorter?)
        # 1st layer must have input_shape parameter
        model.add(Lambda(vgg_preprocess, input_shape=(3,224,224), output_shape=(3,224,224))) 

        self.ConvBlock(2, 64) 
        self.ConvBlock(2, 128)
        self.ConvBlock(3, 256)
        self.ConvBlock(3, 512)
        self.ConvBlock(3, 512) # total of 13 conv layers

        model.add(Flatten())
        self.FCBlock() # with dropout 0.5
        self.FCBlock()
        model.add(Dense(1000, activation='softmax'))

        fname = 'vgg16.h5'
        # get_file() returns path to downloaded file
        # as per: https://github.com/fchollet/keras/blob/master/keras/utils/data_utils.py
        model.load_weights(get_file(fname, self.FILE_PATH+fname, cache_subdir='models')) # stores at ~/.keras/models

    # wrapper for flow_from_dir, returns a generator
    def get_batches(self, path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8, class_mode='categorical'):
        return gen.flow_from_directory(path, target_size=(224,224),
                class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

    # replaces last layer with new, having num softmax activations
    # makes layers not trainable (hence finetuning)
    def ft(self, num):
        model = self.model # this assignment to make shorter?
        model.pop()
        for layer in model.layers: layer.trainable=False
        model.add(Dense(num, activation='softmax'))
        self.compile()


    def finetune(self, batches):
        # prepares model and makes classes, but no training!
        self.ft(batches.num_class)  # Keras2
        '''
        batches.class_indices returns a dict {class_name: class_index}
        list(iter(batches.class_indices)) should be list of keys?
        same as list(batches.class_indices)?
        '''
        classes = list(iter(batches.class_indices))
        for c in batches.class_indices: # loops through keys of dict
            # appends classname at position of that class integer representation in the model
            classes[batches.class_indices[c]] = c 
        self.classes = classes


    def compile(self, lr=0.001):
        self.model.compile(optimizer=Adam(lr=lr),
                loss='categorical_crossentropy', metrics=['accuracy'])


    # Keras2, fit from data loaded in memory
    def fit_data(self, trn, labels,  val, val_labels,  nb_epoch=1, batch_size=64):
        self.model.fit(trn, labels, epochs=nb_epoch,
                validation_data=(val, val_labels), batch_size=batch_size)


    # Keras2, fit from data generator
    # need to manually calculate steps_per_epoch and validation_steps
    def fit(self, batches, val_batches, batch_size, nb_epoch=1):
        self.model.fit_generator(batches, steps_per_epoch=int(np.ceil(batches.samples/batch_size)), epochs=nb_epoch,
                validation_data=val_batches, validation_steps=int(np.ceil(val_batches.samples/batch_size)))

        
    # Keras2
    def test(self, path, batch_size=8):
        test_batches = self.get_batches(path, shuffle=False, batch_size=batch_size, class_mode=None)
        return test_batches, self.model.predict_generator(test_batches, int(np.ceil(test_batches.samples/batch_size)))
