# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:09:10 2019

@author: -
"""
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from joblib import dump

from random import choices

from pickle import load as pload

from numpy import ones
from numpy import hstack
from numpy import vstack
from numpy import copy as ncopy
from numpy.random import shuffle as nshuffle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D

from cv2 import resize as cvresize

class Teacher:
    """General Class from ML"""
    def __init__(self, logger):
        self.logger = logger
        
    def resize(self, array, new_size):
        """Resize a 2D array"""
        try:
            if type(new_size) == int:
                new_size = (new_size, new_size)
                
            return(cvresize(array, new_size))
        except Exception as e:
            self.logger.log(e)
        
    def shuffle(self, array):
        """Suffle a list"""
        try:
            a = ncopy(array)
            nshuffle(a)
            return(a)
        except Exception as e:
            self.logger.log(e)
        
    def random_choices(self, array, number):
        """Choose randomly from a list"""
        try:
            return(choices(array, k=number))
        except Exception as e:
            self.logger.log(e)
            
    def class_adder(self, array, cl):
        """Add class identifier to array"""
        try:
            class_array = ones((array.shape[0], 1)) * cl
            return(hstack((array, class_array)))
        except Exception as e:
            self.logger.log(e)
            
    def class_combiner(self, array1, array2):
        """Stack classes to one array"""
        try:
            return(vstack((array1, array2)))
        except Exception as e:
            self.logger.log(e)
            
class SVM(Teacher):
    """Support Vector Machine Class"""
    def tts(self, array, test_size=0.20):
        """Split array to train and test arrays"""
        try:
            ar = array[:,:-1]
            cl = array[:,-1]
            return(train_test_split(ar, cl, test_size=test_size))
        except Exception as e:
            self.logger.log(e)
            
    def classifier(self, train_x, train_y, kernel="linear", gamma='auto'):
        """Create classifier"""
        clf = SVC(kernel=kernel, gamma=gamma)
        return(clf.fit(train_x, train_y))
        
    def predict(self, classfier, data):
        """Predict for a given value list"""
        return(classfier.predict(data))
        
    def save(self, model, filename):
        """Save classifier to file"""
        dump(model, filename)
        
    def load(self, filename):
        """Save classifier from file"""
        return(pload(open(filename, 'rb')))
        
class CNN(Teacher):
    """Convolutional Neural Network Class"""
    def classifier(self, data, activations=['relu', 'relu', 'sigmoid'],
                   filters=256, windows_size=(3, 3), max_pooling_size=(2, 2),
                   loss='binary_crossentropy',
                   optimizer='adam', metrics=['accuracy'],
                   batch_size=32, epochs=10, validation_split=0.3):
        """Create classifier"""
        try:
            train_x = data[:,:-1]
            train_y = data[:,-1]
            model = Sequential()
            for it, activation in enumerate(activations[:-1]):
                if it == 0:
                    model.add(Conv2D(filters, windows_size,
                                     input_shape=train_x.shape[1:]))
                else:
                    model.add(Conv2D(filters, windows_size))
                model.add(Activation(activation))
                model.add(MaxPooling2D(pool_size=max_pooling_size))
                
            model.add(Flatten())
            model.add(Activation(activations[-1]))
            
            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
            
            model.fit(train_x, train_y, batch_size=batch_size,
                      epochs=epochs, validation_split=validation_split)
            
            return(model)
        except Exception as e:
            self.logger.log(e)