#Week 1 Assignment - Linear Regression via Gradient Descent

#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#************************************ Part a ************************************
#Data
df = pd.read_csv('week1.txt') 
df.head()
df.reset_index(level=0, inplace=True)
df.head()

#Feature
X = np.array(df.iloc[:,0])
X = X.reshape(-1,1)

#Target variable
y = np.array(df.iloc[:,1]);
y = y.reshape(-1,1)

#Normalise
y = (y - min(y))/(max(y)-min(y))
X = (X - min(X))/(max(X)-min(X))

#Visualise
plt.plot(X, y, 'o', color='black');
plt.xlabel('X')
plt.ylabel('y')
plt.title('Data')
plt.show()

#*****************************************

#Linear Regression - Gradient descent 
def gradient_descent(iterations, alpha, X, y):

    #Variable Setup 
    m = len(y)
    theta0 = np.random.random(1)
    theta1 = np.random.random(1)
    cost = np.zeros(iterations)

    #Gradient Descent
    for i in range(iterations):
        h = theta0 + theta1*X 
        d0 = -(2*alpha/m)*np.sum(h-y)
        d1 = -(2*alpha/m)*np.sum(np.dot(X.T, (h-y)))
        theta0 = theta0 + d0
        theta1 = theta1 + d1
        cost[i] = (1/m)*np.sum(np.square(h-y)) 

    return cost, theta0, theta1

#************************************ Part b ************************************

def apply_gradient_descent(iterations, l_rates, X, y):
    ''' Apply gradient descent across a range of learning rates. Plots the resultant cost vs iterations.
    Returns - dataframe of model parameters and cost for all learning rates l_rates '''

    #Set up
    fig = plt.figure(figsize=(15, 10)) 
    count = 0
    df_results = [] 

    for alpha in l_rates:
        
        #Apply gradient descent for given learning rate
        cost, theta0, theta1 = gradient_descent(iterations, alpha, X, y)
        count +=1

        #Plot Cost
        plt.subplot(2, 3, count)
        plt.plot(range(iterations), cost,'b.')
        plt.title('Learning rate: %.3f' %alpha)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')

        #Store model parameters and cost in a dictionary
        d = {
        'l_rate': alpha,  
        'theta0' : '%.5f'%(theta0[0]),
        'theta1' : '%.5f' %(theta1[0]),
        'final cost': '%.5f' %(cost[iterations-1])
        }
        
        df_results.append(d) #, ignore_index=True)
    
    plt.show()
    
    df_results = pd.DataFrame(df_results) #Results; Model parameters and cost across range of learning rates

    return df_results

#**********************************
#Part 1 - Apply gradient descent across a range of learning rates

iterations = 1000
l_rates = [0.005, 0.01, 0.1, 0.2, 0.5, 0.8]
df_results = apply_gradient_descent(iterations, l_rates, X, y)

#*********************************
#Part 2 - Trained model (optimal learning rate)

alpha = 0.1
cost, theta0, theta1 = gradient_descent(iterations, alpha, X, y)

#Plot cost across 200 iterations (zoom in)
iterations = 200
cost, theta0, theta1 = gradient_descent(iterations, alpha, X, y)

#Plot Cost
plt.plot(range(iterations), cost,'r.')
plt.title('Learning rate: %.3f' %alpha)
plt.xlabel('Iterations')
plt.ylabel('Cost')

#Plot trained model vs training data
plt.plot(X, y, 'o', color='black', label = 'training data');
plt.plot(X, y_grad, color = 'red', linewidth= 3.0, label = 'trained model')
plt.legend(loc="upper left")
plt.xlabel('X')
plt.ylabel('y')
plt.title('Data')
plt.show()

#*********************************
#Part 3 - Baseline model

#Cost of baseline model
def baseline_model_cost(theta0, y):
    'Cost of baseline model'
    
    #Variable Setup 
    m = len(y)
    cost = (1/m)*np.sum(np.square(theta0-y)) 

    return cost

theta0_baseline = 0.5
cost_baseline = baseline_model_cost(theta0_baseline, y)

#Plot trained model vs baseline vs data
y_baseline = theta0_baseline*np.ones(len(X))

plt.plot(X, y, 'o', color='black', label = 'training data');
plt.plot(X, y_grad, color = 'red', linewidth= 3.0, label = 'trained model')
plt.plot(X, y_baseline, color = 'blue', linewidth= 3.0, label = 'baseline model')
plt.legend(loc="upper left")
plt.xlabel('X')
plt.ylabel('y')
plt.title('Data')
plt.show()

#*********************************
#Part 4 - Linear Regression - sklearn
reg = LinearRegression().fit(X, y)
reg.coef_
reg.intercept_

