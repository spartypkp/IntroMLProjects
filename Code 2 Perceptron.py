#!/usr/bin/env python
# coding: utf-8

# # Code 2: Perceptron
# 
# ## Part 1: Scikit-Learn Overview
# Scikit-Learn uses the same design principles across its API. This is a brief overview of the format. 
# 
# ### Consistency
# All objects share a consistent and simple interface.
# 
# ### Estimators
# Any object that estimates parameters based on a dataset is called an *estimator*. The estimation is done by the *fit( )* method which takes a dataset (and labels for a supervised algorithm) as a parameter. All other parameters used to guide the estimation are called *hyperparameters*, and these are set as instance variables via a constructor parameter. 
# 
# ### Transformers
# Estimators that can also transform the dataset are called *transformers* (not to be confused with the newest Neural Net Transformer). This is done with the *transform( )* method on the dataset. All transformers also have the *fit_transform( )* method: this is equivalent to calling fit( ) and then transform( ), but is usually optimized to run faster. 
# 
# ### Predictors
# Estimators that are given a dataset and can make predictions on it are called *predictors*. This is done with the *predict( )* method which takes a dataset and returns corresponding predictions. These also have a *score( )* method that measures the quality of the predictions using a test set (and corresponding labels if supervised). 
# 
# ### Inspection
# All the estimator's hyperparameters are accesible through public instance variables (e.g., imputer.strategy) and its learned parameters are accessible through public instance variables with an underscore suffix (e.g., imputer.statistics_). 
# 
# ### Use of other packages
# Scikit-Learn uses other packages rather than create its own classes. Datasets are represented as NumPy arrays or SciPy matrices. Hyperparameters are Python strings or numbers. 
# 
# ### Default Parameters
# Scikit-Learn starts with typical default values for most parameters. 

# ## Part 2: Basic Perceptron
# Let's code a perceptron using Scikit-Learn!
# Your job is to consult the Scikit-Learn API and fill in the skeleton code. 




# imports
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
import os
import tarfile
import urllib
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error





# Load the iris dataset
# It consists of 3 different types of irises’ (Setosa, Versicolour, & Virginica) 
#     petal & sepal length, stored in a 150x4 numpy.ndarray.
#     Rows: samples 
#     Columns: Sepal Length, Sepal Width, Petal Length & Petal Width.
iris = load_iris()
iris["data"]





# Split the dataset into X (features) and y (labels)
# X: should contain all rows, but only the petal length & petal width features
X = iris["data"][:,2:4]

# y: Setosa is our target
y = iris["target"]

print(X)
print(y)





# Create an instance of a Perceptron classifier
per_clf = Perceptron()
per_clf





# Use the Perceptron's fit method on your smaller dataset
per_clf.fit(X, y)





# Use the Perceptron's predict method on sample[2,0.5]
y_pred = per_clf.predict([[2,0.5]])

print(y_pred)


# ## Part 3: Another Perceptron




# import Scikit-Learn's StandardScaler, train_test_split, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score





# Save the iris dataset data and target values to X and y, respectively
X = iris["data"][:,2:4]
y = iris["target"]





# Print/view the first 5 observations from y
print(y[0:5])





# Print/view the first 10 observations of X
print(X[0:10])





# Split the dataset into 80% training and 20% testing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2)





# Train the StandardScaler to format the X training set
# Remember StandardScaler standardizes all features to have
#     mean = 0 and unit variance
sc = StandardScaler()
sc.fit(X_train)





# Apply the StandardScaler to the X training dataset
X_train_std = sc.transform(X_train)

# Apply StandardScaler to X test data
X_test_std = sc.transform(X_test)





# Create a Perceptron object with parameters:
#     50 iterations (epochs) over the dataset
#     learning rate n = 0.1
#     random_state = 0

per = Perceptron(max_iter=50, alpha=0.1, random_state=0)

# Train the Perceptron on the standardized X training set
per.fit(X_train_std, y_train)





# Apply trained Perceptron on standardized X test dataset to make predictions for y
y_pred = per.predict(X_test_std)

# Print predictions
print(y_pred)





# Print true labels
print(y_test)





# Print the accuracy_score of the model 
# You should compare the actual labels with the predicted labels
accuracy_score = per.score(X_test_std , y_test)
print("Accuracy: {:.2f}".format(accuracy_score))







