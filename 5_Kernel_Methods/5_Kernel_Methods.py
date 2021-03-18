# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 10:46:11 2020

@author: Hannah Craddock
"""
#Imports

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pylab as plt

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

plt.rc('font', size=16);
%matplotlib qt


#**********************
#Part 1: Dummy
#Data
Xtrain = np.array([-1, 0, 1]).reshape(-1, 1)
ytrain = [0,1,0]

#**************************************
#Part a
#Function plot for varying gamma 
def plot_knn_regressor(Xtrain, ytrain, k, range_preds, range_gamma, plot_dims):
    'Plot knn predictions for varying parameter gamma in kernelised knn'
    #Setup
    fig = plt.figure(figsize=(20, 20)) 
    count = 0
    #Generate test data
    Xtest=np.linspace(range_preds[0], range_preds[1], num=1000).reshape(-1, 1)
    
    for gammaX in range_gamma:
        
        #Set up gaussian kernel function
        def gaussian_kernel(distances):
            weights = np.exp(-gammaX*(distances**2))
            return weights/np.sum(weights)

        #Model
        model = KNeighborsRegressor(n_neighbors= k, weights=gaussian_kernel).fit(Xtrain, ytrain)
        #Predictions
        ypred = model.predict(Xtest)     
        #Plot
        count +=1
        plt.subplot(plot_dims[0], plot_dims[1], count)
        
        plt.scatter(Xtrain, ytrain, color= 'red', marker='+', linewidth = 3)
        plt.plot(Xtest, ypred, 'g-',linewidth = 3)
        plt.xlabel('input x'); plt.ylabel('output y')
        plt.legend(['kNN','train'])
        plt.title('knn kernelised regression, gamma = {}'.format(gammaX))
        plt.show()
        


#Apply function
k = 3
range_preds = [-3, 3]
range_gamma = [0, 1, 5, 10, 25]
plot_dims = [2, 3] 
#Apply
plot_knn_regressor(Xtrain, ytrain, k, range_preds, range_gamma, plot_dims)  

#**************************************
#Part c - Kernelised Ridge Regression

def plot_kernel_ridge(Xtrain, ytrain, C, range_preds, range_gamma, plot_dims, legend_loc):
    'Plot kernel ridge regression predictions for varying the parameter gamma in the kernel and return coefficients in a dataframe'
    
    #Setup
    fig = plt.figure(figsize=(20, 20)) 
    results = []
    count = 0
    
    #Generate test data
    Xtest=np.linspace(range_preds[0], range_preds[1], num=1000).reshape(-1, 1)
    
    #Test range of gamma values 
    for gammaX in range_gamma:
        count += 1
        
        #Model
        model = KernelRidge(alpha=1.0/2*C, kernel= 'rbf', gamma=gammaX).fit(Xtrain, ytrain)
        
        #Predictions
        ypred = model.predict(Xtest) 
        
        #Coefficients
        d = {
        'gamma' : gammaX,
        'coefficients' :  np.around(model.dual_coef_, decimals = 3),
        }
        
        results.append(d) 

        #Plot
        plt.subplot(plot_dims[0], plot_dims[1], count)
        
        plt.scatter(Xtrain, ytrain, color= 'red', marker='+', linewidth = 3)
        plt.plot(Xtest, ypred, 'g-',linewidth = 3)
        plt.xlabel('input x'); plt.ylabel('output y')
        plt.legend(['Kernel Ridge Regression','train'], loc = legend_loc)
        plt.title('Kernelised Ridge Regression, gamma = {}'.format(gammaX))
        plt.show()
        
        #Return results
        df_coeffs = pd.DataFrame(results)
        
    return df_coeffs

#To do
range_gamma = [0, 1, 5, 10, 25]
C = 0.1
legend_loc = 'center right'
df_coeffs = plot_kernel_ridge(Xtrain, ytrain, C, range_preds, range_gamma, plot_dims, legend_loc)


#***********************************************************************************************************************************************
#Part 2: Dataset
data = pd.read_csv('data_week6.txt')
data.reset_index(inplace=True)
X = np.array(data.iloc[:,0]).reshape(-1, 1) #
y = np.array(data.iloc[:,1]).reshape(-1, 1) 

#Inspect
max(X) #1
min(X) #-1 

#Part a - kernelised knn
#Apply function
k = len(X)
range_preds = [-3, 3] #M
range_gamma = [0, 1, 5, 10, 25]
plot_dims = [2, 3] 
#Apply
plot_knn_regressor(X, y, k, range_preds, range_gamma, plot_dims)  

#Part b - kernelised ridge
C = 0.1
legend_loc = 'upper right'
df_coeffs = plot_kernel_ridge(X, y, C, range_preds, range_gamma, plot_dims, legend_loc)


#************************************************************************************
#Part c Cross validation

def choose_gamma_knn(X, y, range_gamma, plot_color):
    '''Implement 5 fold cv to determine optimal gamma'''
    
    #Param setup
    kf = KFold(n_splits = 5)
    mean_error=[]; std_error=[];
    
    for gammaX in range_gamma:
        #Params
        mse_temp = []
        
        #Set up gaussian kernel function
        def gaussian_kernel(distances):
            weights = np.exp(-gammaX*(distances**2))
            return weights/np.sum(weights)     
                    
        for train, test in kf.split(X):
            #Model
            model = KNeighborsRegressor(n_neighbors= len(train), weights=gaussian_kernel)
            model.fit(X[train], y[train])
            ypred = model.predict(X[test])
            mse = mean_squared_error(y[test], ypred)
            mse_temp.append(mse)
            
        #Get mean & variance
        mean_error.append(np.array(mse_temp).mean())
        std_error.append(np.array(mse_temp).std())
        
    #Plot
    fig = plt.figure(figsize=(15,12))
    plt.errorbar(range_gamma, mean_error, yerr=std_error, color = plot_color)
    plt.xlabel('gamma')
    plt.ylabel('Mean square error')
    plt.title('Choice of gamma in kernelised knn - 5 fold CV')
    plt.show()
    
#Apply
k = len(X)
range_gamma = [0, 1, 5, 10, 15, 20, 25, 40, 50]
plot_color = 'green'
choose_gamma_knn(X, y, range_gamma, plot_color)

#**********************
#Apply optimal model - optimal gamma 
#Data
Xtest=np.linspace(range_preds[0], range_preds[1], num=1000).reshape(-1, 1)

#Gaussian kernel
gammaX = 25
def gaussian_kernel(distances):
    weights = np.exp(-gammaX*(distances**2))
    return weights/np.sum(weights)  

#Model         
model = KNeighborsRegressor(n_neighbors= len(X), weights=gaussian_kernel)
model.fit(X, y)

#Predictions 
predictions = model.predict(Xtest)

#Plot
fig = plt.figure(figsize=(15,12))
plt.scatter(X, y, color= 'red', marker='+', linewidth = 3)
plt.plot(Xtest, predictions, 'g*',linewidth = 3)
plt.xlabel('input x'); plt.ylabel('output y')
plt.legend(['kNN predictions','data'])
plt.title('Optimal knn kernelised regression, gamma = {}'.format(gammaX))
plt.show()

#************************************************************************************
#Part c Cross validation - kernelised Ridge Regression

def choose_gamma_ridge(X, y, range_gamma, alphaX, plot_color):
    '''Implement 5 fold cv to determine optimal gamma'''
    
    #Param setup
    kf = KFold(n_splits = 5)
    mean_error=[]; std_error=[];
    
    for gammaX in range_gamma:
        #Params
        mse_temp = []
        #Model
        model = KernelRidge(alpha= alphaX, kernel= 'rbf', gamma=gammaX)    
        
        #5 fold CV           
        for train, test in kf.split(X):
            #Model
            model.fit(X[train], y[train])
            ypred = model.predict(X[test])
            mse = mean_squared_error(y[test], ypred)
            mse_temp.append(mse)
            
        #Get mean & variance
        mean_error.append(np.array(mse_temp).mean())
        std_error.append(np.array(mse_temp).std())
        
    #Plot
    fig = plt.figure(figsize=(15,12))
    plt.errorbar(range_gamma, mean_error, yerr=std_error, color = plot_color)
    plt.xlabel('gamma')
    plt.ylabel('Mean square error')
    plt.title('Choice of gamma in kernelised Ridge Regression - 5 fold CV, alpha = {}'.format(alphaX))
    plt.show()

#Apply
alphaX = 1
range_gamma = [ 0, 1, 5, 10, 15, 20, 30]
choose_gamma_alpha_ridge(X, y, range_gamma, alphaX, plot_color)


#***************************
def choose_alpha_ridge(X, y, range_C, gammaX, plot_color):
    '''Implement 5 fold cv to determine optimal gamma'''
    
    #Param setup
    kf = KFold(n_splits = 5)
    mean_error=[]; std_error=[];
    
    for C in range_C:
        #Params
        mse_temp = []
        #Model
        model = KernelRidge(alpha= 1.0/(2*C), kernel= 'rbf', gamma=gammaX)    
        
        #5 fold CV           
        for train, test in kf.split(X):
            #Model
            model.fit(X[train], y[train])
            ypred = model.predict(X[test])
            mse = mean_squared_error(y[test], ypred)
            mse_temp.append(mse)
            
        #Get mean & variance
        mean_error.append(np.array(mse_temp).mean())
        std_error.append(np.array(mse_temp).std())
        
    #Plot
    fig = plt.figure(figsize=(15,12))
    plt.errorbar(range_C, mean_error, yerr=std_error, color = plot_color)
    plt.xlabel('C')
    plt.ylabel('Mean square error')
    plt.title('Choice of C in kernelised Ridge Regression - 5 fold CV, gamma = {}'.format(gammaX))
    plt.show()

#Apply
gammaX = 5
range_C = [0.001, 0.01, 1, 5, 10]# 100]
choose_alpha_ridge(X, y, range_C, gammaX, plot_color)

#*****************
#Optimal model
#Data
#Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33, random_state=42)
Xtest=np.linspace(range_preds[0], range_preds[1], num=1000).reshape(-1, 1)

#Model   
C = 1
gammaX = 5
model = KernelRidge(alpha= 1.0/(2*C), kernel= 'rbf', gamma=gammaX)   
model.fit(X, y)
#Params
model.dual_coef_
#Predictions 
predictions = model.predict(Xtest)

#Plot
fig = plt.figure(figsize=(15,12))
plt.scatter(X, y, color= 'red', marker='+', linewidth = 3)
plt.plot(Xtest, predictions, 'g*',linewidth = 3)
plt.xlabel('input x'); plt.ylabel('output y')
plt.legend(['kNN predictions','data'])
plt.title('Optimal Kernelised Ridge Regression, gamma = {}, C = {}'.format(gammaX, C))
plt.show()
