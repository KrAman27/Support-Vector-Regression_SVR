# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 18:23:34 2020

@author: aman kumar
"""
"""CAR SALES PREDICTION
Dataset Description
The dataset contains certain features/info about a customer and his/her car purchasing amount.
Split the dataset suitably into a test and training set. Build a SVR model on it."""

#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv('Car_Purchasing_Data (1).csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, [8]].values

#training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)
X_test = sc_X.transform(X_test)

#fitting the SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X_train,y_train)

#Predicting a new result
y_Pred=sc_y.inverse_transform(regressor.predict(X_test))   
