# -*- coding: utf-8 -*-
"""
Created on Fri May  8 14:59:56 2020

Artificial neural Network
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])
ct = ColumnTransformer([("Geography", OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)
X = X[: , 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
Sc_X = StandardScaler() 
X_train = Sc_X.fit_transform(X_train)
X_test = Sc_X.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers.core import Dense

classifier = Sequential()

classifier.add(Dense(output_dim = 6,
                     init = 'uniform', 
                     activation = 'relu',
                     input_dim = 11))

classifier.add(Dense(output_dim = 6,
                     init = 'uniform', 
                     activation = 'relu'))

classifier.add(Dense(output_dim = 1,
                     init = 'uniform', 
                     activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#fitting the ANN to the training set
classifier.fit(X_train, Y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

