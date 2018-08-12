# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 15:13:46 2018

@author: verma
"""
# Electricity Load Forecasting for State Maine
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

# Importing the dataset
dataset = pd.read_csv('original data.csv')

# coverting 'Date' to date time object
dataset['Date'] = pd.to_datetime(dataset['Date'])

# extracting year, month, day of week from the 'Date' field
dataset['Month']=dataset['Date'].dt.month
dataset['Day']=dataset['Date'].dt.day

# adding column 't2', 't3' to dataset
# t2 = DryBulb * DryBulb
dataset['t2'] = dataset['DryBulb'] * dataset['DryBulb']

# t3 = DryBulb * DryBulb * DryBulb
dataset['t3'] = dataset['DryBulb'] * dataset['DryBulb'] * dataset['DryBulb']

# rearranging dataset
dataset = dataset [['Date', 'Hour', 'DA_DEMD', 'DA_LMP', 'DA_EC', 'DA_CC', 'DA_MLC', 'RT_LMP', 'RT_CC', 'RT_MLC', 'DryBulb', 'DewPnt', 'SM', 'trend', 'Month', 'Day', 't2', 't3', 'DEMAND']]

# dropping columns
dataset = dataset.drop(['DA_DEMD', 'DA_LMP', 'DA_EC', 'DA_CC', 'DA_MLC', 'RT_LMP', 'RT_CC', 'RT_MLC', 'DewPnt'],1)


# separating features and target variables
X = dataset.iloc[:, 1:9].values
y = dataset.iloc[:, 9].values

# using multiple linear regression model
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# calculating mean square error percentage
error = (abs(y_pred) - y_test)/y_test
mean_square_preccentage_error = sum(error)/ len(y_pred)

# using artificial neural network model

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
