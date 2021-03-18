#Week 2 Assignment - Logistic Regression & SVC

#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import cmath 

#*********************** Part a - Data & Logistic Regression *********************************

#***************************
#Part i Data + Visualisation
#Data
data = pd.read_csv('week2.txt',)
data.head()
data.reset_index(inplace=True)
data.columns = ['X1', 'X2', 'y']

#Classe balance
data.y.value_counts()

#Make a copy
df = data.copy()
df.head()

#Extract Features
X1 = df.iloc[:,0]
X2 = df.iloc[:,1]
X = np.column_stack((X1,X2))
y= df.iloc[:,2]

#Plot data and color code the observations according to their value of the target variable y
plt.scatter(df.loc[df['y'] == 1, 'X1'], df.loc[df['y'] == 1, 'X2'], marker = '+', c = 'g')
plt.scatter(df.loc[df['y'] == -1, 'X1'], df.loc[df['y'] == -1, 'X2'], marker = '+', c = 'r')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Data & Logistic Regression model')
plt.legend(['training data, y=1','training data, y=-1'], fancybox=True, framealpha=1) #:D 
plt.show()

#******************* Logistic Regression Model  *********************************

#Part ii Logistic Regression Model

log_reg_model = LogisticRegression(penalty= 'none',solver= 'lbfgs')
log_reg_model.fit(X, y)
log_reg_model.intercept_
log_reg_model.coef_

#*********************************
#Part iii Predictions

predictions = log_reg_model.predict(X)
df['preds'] = predictions 
df.head()

#Test decision boundary
decision_boundary = log_reg_model.intercept_[0] + log_reg_model.coef_[0,0]*X1 + log_reg_model.coef_[0,1]*X2

#Function to plot data, predictions, decision boundary
def plot_data_preds_db (df, log_reg_model):
    'Plot data, logistic regression predictions and decision boundary'
    
    #Get preds
    
    #Decision boundary
    decision_boundary = (-log_reg_model.coef_[0,0]/log_reg_model.coef_[0,1])*X1 - log_reg_model.intercept_[0]/log_reg_model.coef_[0,1]
    
    #Plot of Predictions
    plt.scatter(df.loc[df['preds'] == 1, 'X1'], df.loc[df['preds'] == 1, 'X2'], marker = 'o', facecolors='none', edgecolors= 'k')
    plt.scatter(df.loc[df['preds'] == -1, 'X1'], df.loc[df['preds'] == -1, 'X2'], marker = 'o',facecolors='none', edgecolors= 'y')
    #Plot of Training Data 
    plt.scatter(df.loc[df['y'] == 1, 'X1'], df.loc[df['y'] == 1, 'X2'], marker = '+', c = 'g')
    plt.scatter(df.loc[df['y'] == -1, 'X1'], df.loc[df['y'] == -1, 'X2'], marker = '+', c = 'r')
    #Plot decision boundary
    plt.plot(X1, decision_boundary, linewidth = '4')
    #Labels
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Data & Logistic Regression model')
    plt.legend(['decision boundary', 'predictions, y = 1', 'predictions, y = -1', 'training data, y = 1','training data, y = -1'], fancybox=True, framealpha=1, bbox_to_anchor=(1.04,1), loc="upper left") #:D 
    plt.show()
    
    
#Implement
plot_data_preds_db (df, log_reg_model)


#******************************************* Part b  - SVC ************************************

#SVC - test 
model = LinearSVC(C=1).fit(X, y)
model.intercept_
model.coef_

#Part (i) Test range of values of C parameter

#Range of C
c_test = np.geomspace(0.001, 1000, num = 7)
c_test = np.concatenate((c_test, c_test/2))
c_test = np.sort(c_test)

#Test SVC for range of values of C
def svc_range_c(X, y, c_test):
    '''Implement Support Vector Classification (SVC) 
    for a range of values of the penatly term C '''
    
    #Setup
    df_results = []    
    
    #Loop through c parameters and implement SVC
    for c_param in c_test:
        model = LinearSVC(C= c_param).fit(X, y)
       
        #Dictionarry of values
        d = {
            'C' : c_param,
        'theta0': model.intercept_[0],  
        'theta1' :  model.coef_[0,0],
        'theta2' :  model.coef_[0,1] ,
        }
        
        df_results.append(d) 
    
    #Return dataframe of results - model parameters for a range of penalty terms C
    df_svc_results = pd.DataFrame(df_results) #Results; 
    
    return df_svc_results

#Implement & get dataframe
df_svc_res = svc_range_c(X, y, c_test)

#******************************************************
#Part (ii) Plot Data, Predictions & Decision Boundary

#Function to plot range
def svc_plot_range_c(data, c_test, plot_dim):
    '''Implement Support Vector Classification (SVC) 
    for a range of values of the penatly term C. Plot the resultant SVC predictions and decision boundaries against the data  '''
    
    #Param setup
    fig = plt.figure(figsize=(15, 10)) 
    count = 0
    data_results = [] 
    
    #Features 
    X1 = data.iloc[:,0]
    X2 = data.iloc[:,1]
    X = np.column_stack((X1,X2))
    y= data.iloc[:,2]
        
    
    #Loop through c parameters and implement SVC
    for c_param in c_test:
        count +=1
        model = LinearSVC(C = c_param).fit(X, y)
        #Predictions
        predictions = model.predict(X)
        data['preds'] = predictions
        
        #Plot
        plt.subplot(plot_dim[0], plot_dim[1], count)
        
        plt.scatter(data.loc[data['preds'] == 1, 'X1'], data.loc[data['preds'] == 1, 'X2'], marker = 'o', facecolors='none', edgecolors= 'k')
        plt.scatter(data.loc[data['preds'] == -1, 'X1'], data.loc[data['preds'] == -1, 'X2'], marker = 'o',facecolors='none', edgecolors= 'y')
        #Truth
        plt.scatter(data.loc[data['y'] == 1, 'X1'], data.loc[data['y'] == 1, 'X2'], marker = '+', c = 'g')
        plt.scatter(data.loc[data['y'] == -1, 'X1'], data.loc[data['y'] == -1, 'X2'], marker = '+', c = 'r')
        
        #Decision boundary
        decision_boundary = (-model.coef_[0,0]/model.coef_[0,1])*X1 - model.intercept_[0]/model.coef_[0,1] 
        plt.plot(X1, decision_boundary, linewidth = '4')
        #Labels
        plt.title('SVM, C = %.3f' %c_param)
        plt.xlabel('X1')
        plt.ylabel('X2')

    plt.legend(['decision boundary', 'predictions, y = 1', 'predictions, y = -1', 'training data, y = 1','training data, y = -1'], fancybox=True, framealpha=1, bbox_to_anchor=(1.04,1), loc="upper left") #:D 
    plt.show()    

#Implement function
c_test = np.geomspace(0.001, 1000, num = 7) # Range of c values
plot_dim = [3,3] #plot dimensions
svc_plot_range_c(data, c_test, plot_dim) #implement

#Implement - focus on two values of C
c_test = [0.001, 100]
plot_dim = [1,2]
svc_plot_range_c(data, c_test, plot_dim)

#******************************* Part c  - Logistic Regression + Additional Features ***********************************
#Part (i)
#Data
df2 = data.copy()
df2['X1_sq'] = df2['X1']**2
df2['X2_sq'] = df2['X2']**2
df2.head()

#Features
X1_sq = df2.iloc[:,3]
X2_sq = df2.iloc[:,4]
X_v2 = np.column_stack((X,X1_sq, X2_sq))

#Log reg model
log_reg_model2 = LogisticRegression(penalty= 'none',solver= 'lbfgs')
log_reg_model2.fit(X_v2, y)
log_reg_model2.intercept_
log_reg_model2.coef_

#**********************
#Part ii - Predicitions
predictions2  = log_reg_model2.predict(X_v2)
preds_col = 'preds'
df2[preds_col] = predictions2 

#Function to plot predictions
def plot_data_preds(df, log_reg_model, model_name, preds_col):
    'Plot data, logistic regression predictions'
    
    #Plot of Predictions
    plt.scatter(df.loc[df[preds_col] == 1, 'X1'], df.loc[df[preds_col] == 1, 'X2'], marker = 'o', facecolors='none', edgecolors= 'k')
    plt.scatter(df.loc[df[preds_col] == -1, 'X1'], df.loc[df[preds_col] == -1, 'X2'], marker = 'o',facecolors='none', edgecolors= 'y')
    #Plot of Training Data 
    plt.scatter(df.loc[df['y'] == 1, 'X1'], df.loc[df['y'] == 1, 'X2'], marker = '+', c = 'g')
    plt.scatter(df.loc[df['y'] == -1, 'X1'], df.loc[df['y'] == -1, 'X2'], marker = '+', c = 'r')

    #Labels
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Data & {}'.format(model_name))
    plt.legend(['predictions, y = 1', 'predictions, y = -1', 'training data, y = 1','training data, y = -1'], fancybox=True, framealpha=1, bbox_to_anchor=(1.04,1), loc="upper left") #:D 
    plt.show()

#Implement
preds_col = 'preds'
model_name = 'Logistic Regression model w/ Squared Featues'
plot_data_preds(df2, log_reg_model2, model_name, preds_col)

#******************************************************************
#Part iii - Baseline Model

#Recheck classes
df2.y.value_counts()

#Preds Baseline
preds_col_bl = 'preds_baseline'
df2['preds_baseline'] = np.ones(len(y))

#Plot Baseline model - use plot_data_preds function
preds_col_bl = 'preds_baseline'
model_name2 = 'Baseline model'
plot_data_preds(df2, log_reg_model2, model_name2, preds_col_bl)


#******************************************************************
#Part iii - Quadratic Decision Boundary 
def get_roots(c, log_reg_model):
    '''Get roots of quadratic equation '''
    #
    a = log_reg_model.coef_[0,3]
    b = log_reg_model.coef_[0,1]
    distance = (b**2) - (4 * a*c) 
  
    # find two results 
    root1 = (-b-cmath.sqrt(distance))/(2 * a) 
    root2 = (-b + cmath.sqrt(distance))/(2 * a) 

    #Choose appropriate root
    if ((root1 < 1.1) and (root1 > -0.15)):
        root = root1
    else:
        root = root2
       
    return root

def get_quadratic_dec_bound(X1, log_reg_model2):
    ''' Get decision boundary from logistic regression model with quadratic terms'''
    #Set up
    dec_boundary2 = []
    
    #Loop through all x values and determine corresponding x2 value
    for xx1 in X1:
        c = log_reg_model2.intercept_ + log_reg_model2.coef_[0,0]*xx1 + log_reg_model2.coef_[0,2]*xx1**2 
        xx2 = get_roots(c, log_reg_model2)
        dec_boundary2.append(xx2)
    
    return dec_boundary2

#Implement
dec_boundary2 = get_quadratic_dec_bound(X1, log_reg_model2)

#Plot
def plot_data_preds_dbII(df, log_reg_model, model_name, preds_col, decision_boundary):
    'Plot data, logistic regression predictions + dec_boundary'
    
    #Plot of Predictions
    plt.scatter(df.loc[df[preds_col] == 1, 'X1'], df.loc[df[preds_col] == 1, 'X2'], marker = 'o', facecolors='none', edgecolors= 'k')
    plt.scatter(df.loc[df[preds_col] == -1, 'X1'], df.loc[df[preds_col] == -1, 'X2'], marker = 'o',facecolors='none', edgecolors= 'y')
    #Plot of Training Data 
    plt.scatter(df.loc[df['y'] == 1, 'X1'], df.loc[df['y'] == 1, 'X2'], marker = '+', c = 'g')
    plt.scatter(df.loc[df['y'] == -1, 'X1'], df.loc[df['y'] == -1, 'X2'], marker = '+', c = 'r')
    
    #Decision boundary
    X1 = df.iloc[:,0]
    plt.scatter(X1, decision_boundary, marker = '*', linewidth = '1')
    
    #Labels
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Data & {}'.format(model_name))
    plt.legend(['predictions, y = 1', 'predictions, y = -1', 'training data, y = 1','training data, y = -1', 'decision boundary'], fancybox=True, framealpha=1, bbox_to_anchor=(1.04,1), loc="upper left") #:D 
    plt.show()
    
#Implement
preds_col = 'preds'
model_name = 'Logistic Regression model w/ Squared Featues'
plot_data_preds_dbII(df2, log_reg_model2, model_name, preds_col, dec_boundary2)




