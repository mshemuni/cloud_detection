# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:09:10 2019

@author: -
"""
import datetime


from random import choices

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score as ascore
from sklearn.metrics import balanced_accuracy_score as bascore
from sklearn.metrics import average_precision_score as apscore
from sklearn.metrics import f1_score as f1score
from sklearn.metrics import precision_score as pscore
from sklearn.metrics import recall_score as rcscore
from sklearn.metrics import jaccard_score as jcscore
from sklearn.metrics import roc_auc_score as rascore

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import BernoulliNB 
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.callbacks import Callback

from sklearn.svm import SVC

from numpy import ones
from numpy import hstack
from numpy import vstack
from numpy import copy as ncopy
from numpy.random import shuffle as nshuffle

from joblib import load as jlload
from joblib import dump as jldump

class MyCustomCallback(Callback):

  def on_train_batch_begin(self, batch, logs=None):
    print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

  def on_train_batch_end(self, batch, logs=None):
    print('Training: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))

  def on_test_batch_begin(self, batch, logs=None):
    print('Evaluating: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

  def on_test_batch_end(self, batch, logs=None):
    print('Evaluating: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))

class Teacher:
    """General Class from ML"""
    def __init__(self, logger):
        self.logger = logger
        
    def save(self, clsf, file_name):
        """Save given classifier to file"""
        self.logger.log("Saving Classifier")
        try:
            jldump(clsf, file_name) 
        except Exception as excpt:
            self.logger.log(excpt)
            
    def load(self, file_name):
        """Load given classifier from file"""
        self.logger.log("Loading Classifier")
        try:
            return(jlload(file_name))
        except Exception as excpt:
            self.logger.log(excpt)
        
    def accuracy(self, y_test, predict):
        try:
            scores = {"Accuracy classification score": ascore(y_test, predict),
                      "Balanced accuracy": bascore(y_test, predict),
                      "Average precision score": apscore(y_test, predict),
                      "F1 score, also known as balanced F-score or F-measure": f1score(y_test, predict),
                      "The precision": pscore(y_test, predict),
                      "The recall": rcscore(y_test, predict),
                      "Jaccard similarity coefficient score": jcscore(y_test, predict),
                      "Area Under the Receiver Operating Characteristic Curve (ROC AUC) score": rascore(y_test, predict)}
            return(scores)
        except Exception as excpt:
            self.logger.log(excpt)
        
    def tts(self, array, test_size=0.20, shuffle=True):
        """Split array to train and test arrays"""
        self.logger.log("Splitting test and training")
        try:
            return train_test_split(array, test_size=test_size,
                                    shuffle=shuffle)
        except Exception as excpt:
            self.logger.log(excpt)
        
    def shuffle(self, array):
        """Suffle a list"""
        self.logger.log("Shuffling array")
        try:
            new_array = ncopy(array)
            nshuffle(new_array)
            return new_array
        except Exception as excpt:
            self.logger.log(excpt)

    def random_choices(self, array, number):
        """Choose randomly from a list"""
        self.logger.log("Choosing randomly")
        try:
            return choices(array, k=number)
        except Exception as excpt:
            self.logger.log(excpt)

    def class_adder(self, array, cl):
        """Add class identifier to array"""
        self.logger.log("Adding class")
        try:
            class_array = ones((array.shape[0], 1)) * cl
            return hstack((array, class_array))
        except Exception as excpt:
            self.logger.log(excpt)

    def class_combiner(self, array1, array2):
        """Stack classes to one array"""
        self.logger.log("Combining classes")
        try:
            return vstack((array1, array2))
        except Exception as excpt:
            self.logger.log(excpt)
            
    def predict(self, classfier, data):
        """Predict for a given value list"""
        self.logger.log("Predicting")
        try:
            return classfier.predict(data)
        except Exception as excpt:
            self.logger.log(excpt)

class SVM(Teacher):
    """Support Vector Machine Class"""

    def classifier(self, X_train, y_train, kernel="linear", gamma='auto'):
        """Create classifier"""
        try:
            self.logger.log("Training")
            clf = SVC(kernel=kernel, gamma=gamma)

            return clf.fit(X_train, y_train)
        except Exception as excpt:
            self.logger.log(excpt)
            
class CNN(Teacher):
    """Convolutional Neural Network Class"""        
    def __show__(self, clsf):
        try:
            
            plt.plot(clsf.history['accuracy'])
            plt.plot(clsf.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
            
            plt.plot(clsf.history['loss'])
            plt.plot(clsf.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
        except Exception as excpt:
            self.logger.log(excpt)
    
    def classifier(self, X_train, y_train, X_test, y_test, epochs=3, plot=False, evaluate=True):
        try:
            model = Sequential()
            
            model.add(Conv2D(256, (3, 3), input_shape=X_train.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            
            model.add(Conv2D(256, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            
            model.add(Flatten())
            
            model.add(Dense(64))
            
            model.add(Dense(1))
            model.add(Activation('sigmoid'))
            
            model.compile(loss='binary_crossentropy', optimizer='adam',
                          metrics=['accuracy'])
            
            history = model.fit(X_train, y_train, batch_size=32, epochs=epochs,
                            validation_data=(X_test, y_test))
            
            if evaluate:
                evl = model.evaluate(X_test, y_test, batch_size=32,
                                     verbose=1, steps=1,
                                     callbacks=[MyCustomCallback()])
                print(evl)
            if plot:
                self.__show__(history)
            return(model)
        except Exception as excpt:
            self.logger.log(excpt)

class KNN(Teacher):
    def classifier(self, X_train, y_train, n_neighbors=3):
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        return knn.fit(X_train, y_train)
    
class LR(Teacher):
    def classifier(self, X_train, y_train, max_iter=10000):
        try:
            logisticRegr = LogisticRegression(max_iter=max_iter)
            logisticRegr.fit(X_train, y_train)
            return(logisticRegr)
        except Exception as excpt:
            self.logger.log(excpt)
            
class NB(Teacher):
    def __type_detector__(self, the_type):
        if "BERNOULLI".startswith(the_type.upper()):
            return "BERNOULLI"
        elif "CATEGORICAL".startswith(the_type.upper()):
            return "CATEGORICAL"
        elif "COMPLEMENT".startswith(the_type.upper()):
            return "COMPLEMENT"
        elif "MULTINOMIAL".startswith(the_type.upper()):
            return "MULTINOMIAL"
        else:
            return "GAUSSIAN"
        
    def classifier(self, X_train, y_train, tp="GAUSSIAN"):
        try:
            wanted_type = self.__type_detector__(tp)
            
            model = GaussianNB()
            
            if wanted_type == "BERNOULLI":
                model = BernoulliNB()
            elif wanted_type == "CATEGORICAL":
                model = CategoricalNB()
            elif wanted_type == "COMPLEMENT":
                model = ComplementNB()
            elif wanted_type == "MULTINOMIAL":
                model = MultinomialNB()
                
            model.fit(X_train ,y_train)
            
            return model

        except Exception as excpt:
            self.logger.log(excpt)
        
