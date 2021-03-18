#Week 3 Assignment - Lasso, Ridge Regression & Cross Validation

#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
import matplotlib
from sklearn.metrics import mean_squared_error

#Plot params
plt.rcParams['figure.dpi'] = 150
matplotlib.rc('xtick', labelsize=10) 
matplotlib.rc('ytick', labelsize=10) 
pd.set_option('display.max_colwidth', -1)
MEDIUM_SIZE = 12
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels

#*********************** Part a - Data  ********************

#***************************
#Part i Data + Visualisation
#Data
data = pd.read_csv('week3.txt',)
data.head()
data.reset_index(inplace=True)
data.columns = ['X1', 'X2', 'y']

#Extract Features
X1 = df.iloc[:,0]
X2 = df.iloc[:,1]
X = np.column_stack((X1,X2))
y= df.iloc[:,2]

#Plot data - 3d visualisation
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.set_title('Data')
ax.scatter(X1, X2, y)

#******************* Part b Lasso Regression Model  *********************************

#Get model results
def regression_model_range_c(X, y, degree_poly, c_test, model_type):
    '''Implement regression (lasso or ridge)
    for a range of values of the penatly term C '''
    
    #Setup
    Xpoly = PolynomialFeatures(degree_poly).fit_transform(X)
    df_results = []    
    
    #Loop through c parameters and implement regression (lasso or ridge)
    for c_param in c_test:
        if model_type == 'Lasso':
            model = Lasso(alpha=1/(2*c_param))   
        elif model_type == 'Ridge':
            model = Ridge(alpha=1/(2*c_param))   
        
        #Fit data
        model.fit(Xpoly, y)
       
        #Dictionarry of values
        d = {
            'C' : c_param,
        'intercept': model.intercept_,  
        'coefficients' :  np.around(model.coef_, decimals = 3),
        }
        
        df_results.append(d) 

    df_svc_results = pd.DataFrame(df_results) #Results; 
    
    return df_svc_results

#Apply to Lasso
degree_poly = 5 
c_test = [1, 10, 100, 500, 1000]
model_type = 'Lasso'
regression_model_range_c(X, y, degree_poly, c_test, model_type)

#*********************************
#Part c Predictions

def plot_preds_range_c(X, y, Xtest, c_test, model_type, plot_colors):
    '''Plot predictions from lasso model for a range of C values '''
    
    #Get polynomial featrues
    Xpoly = PolynomialFeatures(degree_poly).fit_transform(X) 
    Xpoly_test = PolynomialFeatures(degree_poly).fit_transform(Xtest)
    
    #Loop through c parameters and implement lasso regression 
    for c_param in c_test:
        
        if model_type == 'Lasso':
            model = Lasso(alpha=1/(2*c_param))   
        elif model_type == 'Ridge':
            model = Ridge(alpha=1/(2*c_param))
            
        #Fit model
        model.fit(Xpoly, y)
        predictions = model.predict(Xpoly_test)
        
        #Plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        #Plot predictions
        ax.plot_trisurf(Xtest[:,0], Xtest[:,1], predictions, color = plot_colors[0], alpha=.5)  
        #Plot Data
        ax.scatter(X[:,0], X[:,1], y, color = plot_colors[1], label = 'Data')
        #Plot configuration
        colors = ['y', 'r']
        ax.set_xlabel('X1')
        ax.set_ylabel('X2') 
        ax.set_zlabel('Y')
        ax.set_title('{}, C = {}'.format(model_type, c_param), fontdict={'fontsize': 8.5})
        #Legend
        scatter1_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=plot_colors[0], marker = 'o')
        scatter2_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=plot_colors[1], marker = 'v')
        ax.legend([scatter1_proxy, scatter2_proxy], ['{} Predictions'.format(model_type), 'Data'], numpoints = 1)
        ax.view_init(azim = 60) 
        
#Apply function to Lasso
model_type = 'Lasso'
plot_colors = ['y', 'r']
c_test = [1, 10, 100, 500, 1000]
plot_preds_range_c(X, y, Xtest, c_test, model_type, plot_colors)


#******************* Part e Ridge Regression  *********************************

#Get Regression results
degree_poly = 5 
c_test = [1, 10, 100, 500, 1000]
model_type = 'Ridge'
regression_model_range_c(X, y, degree_poly, c_test, model_type)

#Plot Predictions 
model_type = 'Lasso'
plot_colors = ['y', 'r']
plot_preds_range_c(X, y, Xtest, c_test, model_type, plot_colors)


#*********** Part 2 Cross Validation ******************************

def get_df_kfold_cv(X, y, degree_poly, folds, C, model_type):
    '''Implement k fold cross validation for testing 
    regression model (lasso or ridge) and return dataframe of results'''
    
    #Param setup
    mean_error=[]; std_error=[]; df_results = [] 
    Xpoly = PolynomialFeatures(degree_poly).fit_transform(X) 
    #Model
    if model_type == 'Lasso':
        model = Lasso(alpha=1/(2*C))   
    elif model_type == 'Ridge':
        model = Ridge(alpha=1/(2*C)) 
    
    #Loop through each k fold
    for k in folds:
        mse_temp = []
        kf = KFold(n_splits = k)
        
        for train, test in kf.split(Xpoly):
            model.fit(Xpoly[train], y[train])
            #Predict on test set
            ypred = model.predict(Xpoly[test])
            mse = mean_squared_error(y[test],ypred)
            mse_temp.append(mse)
        
        #Dictionarry of values - mean & variance                
        d = {
            'k folds' : k,
        'mse results': np.around(mse_temp, decimals = 3),
        'mean mse' :  np.around(np.array(mse_temp).mean(), decimals = 3), 
        'std mse' : np.around(np.array(mse_temp).std(), decimals = 3)
        }
        
        df_results.append(d) 
    df_kf_results = pd.DataFrame(df_results) #Results
    
    return df_kf_results

#Apply to Lasso
c = 10
model_type = 'Lasso'
folds = [2, 5, 10, 15, 25, 50, 100]
df_kf_results = get_df_kfold_cv(X, y, degree_poly, folds, c, model_type)
df_kf_results

#Train and test results
def get_df_kfold_cvII(X, y, degree_poly, folds, C, model_type):
    '''Implement k fold cross validation for testing 
    regression model (lasso or ridge) and return dataframe of results. Test on train and test set'''
    
    #Param setup
    mean_error=[]; std_error=[]; df_results = [] 
    Xpoly = PolynomialFeatures(degree_poly).fit_transform(X) 
    #Model
    if model_type == 'Lasso':
        model = Lasso(alpha=1/(2*C))   
    elif model_type == 'Ridge':
        model = Ridge(alpha=1/(2*C)) 
    
    #Loop through each k fold
    for k in folds:
        mse_train_temp = []
        mse_test_temp = []
        kf = KFold(n_splits = k)
        
        for train, test in kf.split(Xpoly):
            model.fit(Xpoly[train], y[train])
            
            #Predict on train set
            ypred = model.predict(Xpoly[train])
            mse = mean_squared_error(y[train],ypred)
            mse_train_temp.append(mse)
            
            #Predict on test set
            ypred = model.predict(Xpoly[test])
            mse = mean_squared_error(y[test],ypred)
            mse_test_temp.append(mse)
        
        #Dictionarry of values - mean & variance                
        d = {
            'k folds' : k,
        'mse train results': np.around(mse_train_temp, decimals = 3),
        'mean mse train' :  np.around(np.array(mse_train_temp).mean(), decimals = 3), #np.around(model.coef_, decimals = 3),
        'std mse train' : np.around(np.array(mse_train_temp).std(), decimals = 3),
        'mse test': np.around(mse_test_temp, decimals = 3),
        'mean mse test' :  np.around(np.array(mse_test_temp).mean(), decimals = 3), #np.around(model.coef_, decimals = 3),
        'std mse test' : np.around(np.array(mse_test_temp).std(), decimals = 3)
        }
        
        df_results.append(d) 

    df_kf_results = pd.DataFrame(df_results) #Results
    
    return df_kf_results

#Apply
model_type = 'Lasso'
folds = [2, 5, 10, 15, 25, 50, 100]
c = 10
df_kf2 = get_df_kfold_cv2(X, y, degree_poly, folds, c, model_type)

#*** Plot results ********
def plot_kf_cv(df_kf):
    'Plot mean and std of kfold cross validation model results'
    
    #Plot
    plt.errorbar(np.array(df_kf.iloc[:,0]), np.array(df_kf.iloc[:,5]), yerr=df_kf.iloc[:,6], color = 'orange', alpha = 0.8) #,linewidth = 2)
    plt.errorbar(np.array(df_kf.iloc[:,0]), np.array(df_kf.iloc[:,2]), yerr=df_kf.iloc[:,3], color = 'green', alpha = 0.9) #, linewidth = 2)
    plt.xlabel('k fold')
    plt.ylabel('Mean square error')
    plt.title('K fold cross validation - lasso regression')
    plt.legend(['Test set', 'Training set'])
    plt.show()  

#Apply
plot_kf_cv(df_kf2)

#******** Part e C vs MSE using k fold Cross Val *******
def plot_kf_cv_c_vs_mse(X, y, degree_poly, k, c_test, model_type, plot_color):
    '''Implement k fold cross validation for testing 
    regression model (lasso or ridge) and plot results'''
    
    #Param setup
    kf = KFold(n_splits = k)
    mean_error=[]; std_error=[]; df_results = [] 
    Xpoly = PolynomialFeatures(degree_poly).fit_transform(X) 
    
    #Loop through each k fold
    for c in c_test:
        mse_temp = []
        #Model
        if model_type == 'Lasso':
            model = Lasso(alpha=1/(2*c))   
        elif model_type == 'Ridge':
            model = Ridge(alpha=1/(2*c)) 
                
        for train, test in kf.split(Xpoly):
            model.fit(Xpoly[train], y[train])
            ypred = model.predict(Xpoly[test])
            mse = mean_squared_error(y[test],ypred)
            mse_temp.append(mse)
        
        #Get mean & variance
        mean_error.append(np.array(mse_temp).mean())
        std_error.append(np.array(mse_temp).std())
        
    #Plot
    plt.errorbar(c_test, mean_error, yerr=std_error, color = plot_color)
    plt.xlabel('C')
    plt.ylabel('Mean square error')
    plt.title('K fold CV - Choice of C in {} regression'.format(model_type))
    plt.show()

#Apply to Lasso
k = 5
model_type = 'Lasso'
c_test = [1, 5, 10, 50, 100, 500]
plot_color = 'red' 
plot_kf_cv_c_vs_mse(X, y, degree_poly, k, c_test, model_type, plot_color)

#Apply to Ridge 
k = 5 
model_type = 'Ridge'
c_test = [1, 5, 10, 20, 50] #, 100, 500] 
plot_color = 'orange' 
plot_kf_cv_c_vs_mse(X, y, degree_poly, k, c_test, model_type, plot_color)

