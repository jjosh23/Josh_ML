# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 15:19:33 2019
1.The Purpose of the following code is to create a simple Linear Regression
Steps are as follows:
    1.Import the packages 
    2.Read the dataset 
    3.Divide the dataset into X and y axis
    4.Divide the dataset into test and train sets
    5.Train the model with the test set
    6.Plot the model

@author: Joshua
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv("D:\Datascience\KrishNairPOCs\LinearRegression\SalaryDataset.csv")
#Divided Dataset into X axis and Y axis

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

#Splitting the training and test dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
simple_linear_reg=LinearRegression()
simple_linear_reg.fit(X_train,y_train)

y_predict=simple_linear_reg.predict(X_test)

#Implement the graph

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,simple_linear_reg.predict(X_train))
plt.show()