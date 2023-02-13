#!/usr/bin/env python
# coding: utf-8

# <b>1 Write a function to generate an m+1 dimensional data set, of size n, consisting of m continuous independent variables (X) and one dependent variable (Y) defined as<br>
# yi = xi β + e </b><br>
# where <br>
# ● e is a Gaussian distribution with mean 0 and standard deviation (σ), <br>
# representing the unexplained variation in Y <br>
# ● β is a random vector of dimensionality m + 1,<br>
# representing the coefficients of the linear relationship between X and Y,<br>
# and ∀i∈ [1, n], xi0 = 1 <br>
# <b>The function should take the following parameters:</b><br>
# • σ: The spread of noise in the output variable<br>
# • n: The size of the data set<br>
# • m: number of independent variables Output from the function should be:<br>
# • X: An n × m+1 numpy array of independent variable values (with a 1 in the first column)<br>
# • Y: The n × 1 numpy array of output values <br>
# • β: The random coefficients used to generate Y from X.<br>

# In[3]:


import numpy as np
import matplotlib.pyplot as plt


# In[4]:


np.random.seed(8)


# In[5]:


def generate(n, m, sigma):
    x = np.random.rand(n, m+1)
    e = np.random.normal(0, sigma, (n, 1))
    x[:, 0] = 1
    beta = np.random.rand(m + 1, 1)
    y = np.matmul(x, beta) + e
    return x, y, beta


# In[7]:


x, y, beta = generate(2, 4, 10)
print(x)
print(y)
print(beta)


# <b>2 Write a function that learns the parameters of a linear regression line given inputs</b><br>
# • X: An n×mnumpy array of independent variable values<br>
# • Y: The n× 1 numpy array of output values <br>
# • k: the number of interactions (epochs)<br>
# • τ: the threshold on change in Cost function value from the previous to currentiteration <br>
# • λ: the learning rate for Gradient Descent <br>
# The function should implement the Gradient Descent algorithm that initializes β with random values and then updates these values in each interaction by moving in the direction defined by the partial derivative of the cost function with respect to each of the coefficients. The function should use only one loop that ends after a number of iterations (k) or a threshold on the change in cost function value (τ ).<br>
# The output should be an m + 1 dimensional vector of coefficients and the final cost function value.

# In[9]:


#the cost function with respect to each of the coefficients
def cost(pred_y, y, n):
    cost = (1 / (2*n)) * np.sum((pred_y-y)**2)
    return cost


# In[10]:


#gradient function used for initializes βeta with random values 
def gradient(x, y, pred_y, n):
    gradient = (-2/n)*np.matmul((y- pred_y).T, x)#where T is the threshold
    return gradient


# In[11]:


#regression function
def regression(x, y, epochs, threshold):
    learningRate = 0.001
    n = x.shape[0]
    m = x.shape[1]
    weigt = np.random.rand(m, 1)*learningRate
    preCost = float('inf')#initial cost
    
    for itr in range(epochs):
        pred_y = np.matmul(x, weigt)
        k = cost(pred_y, y, n)
        if abs(preCost-k).all()<=threshold:
            break
        preCost = k
        gd = gradient(x, y, pred_y, n)
        weigt = weigt - learningRate * gd
    return preCost, weigt


# In[12]:


finalCostValue, weigt = regression(x, y, 500, 0.002)
finalCostValue


# In[13]:


weigt

