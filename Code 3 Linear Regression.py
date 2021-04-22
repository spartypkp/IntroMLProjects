#!/usr/bin/env python
# coding: utf-8

# # Linear Regression
# 
# ### Recap
# A linear model makes a prediction by computing a weighted sum of the input features and a constant (bias).    
# Basic linear regression model: $\hat{y}$ = $\theta$<sub>0</sub> + $\theta$<sub>1</sub>x<sub>1</sub> + $\theta$<sub>2</sub>x<sub>2</sub> + ... + $\theta$<sub>n</sub>x<sub>n</sub>    
# * $\hat{y}$ is the predicted value   
# * n is the number of features
# * x<sub>i</sub> is the i<sup>th</sup> feature value
# * $\theta$<sub>j</sub> is the j<sup>th</sup> model parameter ($\theta$<sub>0</sub> is the bias term; $\theta$<sub>1</sub> .. $\theta$<sub>n</sub> are the feature weights)   
# 
# Using linear algebra we can work with this equation in its closed form solution (also called normal equation). Recall the closed form solution for linear regression from class: w* = (X<sup>T</sup>X)<sup>-1</sup> X<sup>T</sup>y
# 
# We can think of the weights (w*) as the general parameter, $\theta$: $\hat{\theta}$ = (X<sup>T</sup>X)<sup>-1</sup> X<sup>T</sup>y
# 
# $\hat{\theta}$ is the values of $\theta$ that minimizes the cost function    
# y is the target vector (i.e., labels or actual y values) 
# 

# ### Part 1: Closed Form Solution




# Import packages
import numpy as np
import sklearn





# For plots

import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)





# Create linear-like data to test first equation
X = 2 * np.random.rand(100, 1)

# Function for generating data: y = 4 + 3x_1 + Gaussian noise
# So theta0 = 4 & theta1 = 3
y = 4 + 3 * X + np.random.randn(100, 1)

# Visualize dataset 
plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])
plt.show()


# Add x0 = 1 to each instance
X2 = np.c_[np.ones((100, 1)), X]



# Use linalg inv & dot to calculate the closed form solution with X2 & y
theta_hat = np.linalg.inv((X2.T @ X2)) @ X2.T @ y

print(theta_hat)


# TODO: Answer in the comments
# theta_hat should print values around [4.3, 2.8]-ish
# But theta0 = 4 & theta1 = 3
# Why couldn't we recover the exact parameters of our original function?
# Your answer: There are inherent noise in the data. Therefore, no matter the inputs we cannot get the true f(x) as our predicted output.



# Task of the next 3 cells: Make a prediction using theta_hat
# Create a new dataset (what dimensions should it be?)

X_new = 2 * np.random.rand(100, 1)

# Add x0 = 1 to each instance
X_new2 = np.c_[np.ones((100, 1)), X_new]



# Use your model to make a prediction on your new dataset
y_predict = np.matmul(X_new2, theta_hat)

# Print y_predict
print(y_predict)


# Plot your new model's predictions
# You should see a red line falling mostly in the middle of the blue data points
plt.plot(X_new2[:,1], y_predict , "r-")

# Plots the original data
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show( )


# ### Part 2: Using Scikit-Learn
# Scikit-Learn's LinearRegression class is based on scipy.linalg.lstsq() (Least Squares). 


# Import the LinearRegression class
from sklearn.linear_model import LinearRegression

# Create a LinearRegression instance
lin_reg = LinearRegression()


# Fit your model
# NOTE: there is a bug in Windows for this method
# If you get this error: ValueError: illegal value in 4th argument of internal None
# Go to the previous cell and create your class instance with the argument: normalize = True

lin_reg.fit(X, y)


# Print the intercept of your model
print(lin_reg.intercept_)





# Print the estimated coefficients of your model
print(lin_reg.coef_)


# Use your model to make a prediction on X_new
# Don't need to answer: are your results similar to y_predict?
predict = lin_reg.predict(X_new)
print(predict)


# ### Part3: Stochastic Gradient Descent

# Import the SGDRegressor class
from sklearn.linear_model import SGDRegressor


# Create an SGDRegressor with
# Maximum number of iterations = 1000
# Training stopping criterion of 1e-3 
# Ridge regularization term
# Initial learning rate of 0.001
sgd_reg = SGDRegressor(max_iter = 1000, tol = 1e-3, eta0 = 0.001)



# Fit the model 
sgd_reg.fit(X,y)


# Print the intercept
print(sgd_reg.intercept_)



# Print the estimated coefficients
print(sgd_reg.coef_)


# The intercept and coefficient should be close to the values of $\hat{\theta}$ found by the closed form solution in Part 1. If they're not, you can (optionally) change the regularization and learning rate until you find values that work better. 
