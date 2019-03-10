# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 16:19:24 2019

@author: masha
"""

from keras.layers import Flatten
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
import cv2
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import utils
from keras.models import load_model
from keras.optimizers import SGD
from data_utils import *
import numpy as np

NUM_CLASSES = 43
IMG_SIZE = 32


def preprocess_features(X):
    # convert from RGB to YUV
    X = np.array([np.expand_dims(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YUV)[:, :, 0], 2) for rgb_img in X])
    X = np.array([ele / 255 for ele in X])

    return X

def get_image_generator():
    # create the generator to perform online data augmentation
    image_datagen = ImageDataGenerator(rotation_range=15.)
    return image_datagen

def get_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    ####
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model


def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / 10))


if __name__ == "__main__":
    X_train, y_train = load_traffic_sign_data('data/train.p')
    n_classes = np.unique(y_train).shape[0]
    y_train = utils.to_categorical(y_train, n_classes)
    # Number of examples
    n_train = X_train.shape[0]
    # What's the shape of an traffic sign image?
    image_shape = X_train[0].shape
    # How many classes?

    model = get_model()
    lr = 0.01
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    batch_size = 32
    nb_epoch = 20
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              validation_split=0.2,
              shuffle=True,
              callbacks=[LearningRateScheduler(lr_schedule),
                         ModelCheckpoint('my_model.h5', save_best_only=True)])




