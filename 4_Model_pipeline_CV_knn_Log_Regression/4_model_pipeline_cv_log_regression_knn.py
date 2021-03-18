#Week 4 Assignment - Logistic Regression + knn models
#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

#Plot params
plt.rcParams['figure.dpi'] = 120
matplotlib.rc('xtick', labelsize=10) 
matplotlib.rc('ytick', labelsize=10) 
pd.set_option('display.max_colwidth', -1)
MEDIUM_SIZE = 12
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels

#*********************** Part 1 - Dataset 1  ********************

#***************************
#Part i Data + Visualisation
#Data
data = pd.read_csv('week4_d1.txt',)
data.head()
data.reset_index(inplace=True)
data.columns = ['X1', 'X2', 'y']

#Extract Features
X1 = df.iloc[:,0]
X2 = df.iloc[:,1]
X = np.column_stack((X1,X2))
y= df.iloc[:,2]

#Plot data and color code the observations according to their value of the target variable y
plt.scatter(data.loc[data['y'] == 1, 'X1'], data.loc[data['y'] == 1, 'X2'], marker = '+', c = 'g')
plt.scatter(data.loc[data['y'] == -1, 'X1'], data.loc[data['y'] == -1, 'X2'], marker = '+', c = 'r')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Data')
plt.legend(['training data, y=1','training data, y=-1'], fancybox=True, framealpha=1) #:D 
plt.show()

#******************* Part a Logistic Regression Model  *********************************

#i. Choose q
def choose_q_cv(X, y, q_range, c_param, plot_color):
    '''Implement 5 fold cross validation for testing 
    regression model (lasso or ridge) and plot results'''
    
    #Param setup
    kf = KFold(n_splits = 5)
    mean_error=[]; std_error=[];
    model = LogisticRegression(penalty= 'l2', C = c_param)
    
    #Loop through each k fold
    for q in q_range:
        mse_temp = []
        Xpoly = PolynomialFeatures(q).fit_transform(X) 
                
        for train, test in kf.split(Xpoly):
            model.fit(Xpoly[train], y[train])
            ypred = model.predict(Xpoly[test])
            mse = mean_squared_error(y[test],ypred)
            mse_temp.append(mse)
        
        #Get mean & variance
        mean_error.append(np.array(mse_temp).mean())
        std_error.append(np.array(mse_temp).std())
        
    #Plot
    plt.errorbar(q_range, mean_error, yerr=std_error, color = plot_color)
    plt.xlabel('q')
    plt.ylabel('Mean square error')
    plt.title('Choice of q in logistic regression - 5 fold CV, C = {}'.format(c_param))
    plt.savefig("log_reg_C.png", bbox_inches='tight')
    plt.show()

#Implement
q_range = [1,2,3,4,5,6,7]
plot_color = 'b'
c_param = 1 #0.001, 1000
choose_q_cv(X, y, q_range, c_param, plot_color)

#********************************************
#i. Choose C
def choose_C_cv(X, y, q, c_range, plot_color):
    '''Implement 5 fold cross validation for testing 
    regression model (lasso or ridge) and plot results'''
    
    #Param setup
    kf = KFold(n_splits = 5)
    Xpoly = PolynomialFeatures(q).fit_transform(X) 
    mean_error=[]; std_error=[];
       
    #Loop through each k fold
    for c_param in c_range:
        mse_temp = [] 
        model = LogisticRegression(penalty= 'l2', C = c_param)
                
        for train, test in kf.split(Xpoly):
            
            model.fit(Xpoly[train], y[train])
            ypred = model.predict(Xpoly[test])
            mse = mean_squared_error(y[test],ypred)
            mse_temp.append(mse)
        
        #Get mean & variance
        mean_error.append(np.array(mse_temp).mean())
        std_error.append(np.array(mse_temp).std())
        
    #Plot
    plt.errorbar(c_range, mean_error, yerr=std_error, color = plot_color)
    plt.xlabel('C')
    plt.ylabel('Mean square error')
    plt.title('Choice of C in Logistic regression - 5 fold CV')
    plt.show()
    
#Implement
q = 2 #Optimal q
c_range = [0.01, 0.02, 0.05, 1, 5, 10, 20, 50, 100, 500, 1000]
q_range = [1,2,3,4,5]
plot_color = 'g'
choose_C_cv(X, y, q, c_range, plot_color)

#********************************************
#Final model
#Data for model training
indices = np.arange(0,600)
Xpoly = PolynomialFeatures(2).fit_transform(X) 
Xpoly_train, Xpoly_test, ytrain, ytest,indices_train,indices_test = train_test_split(Xpoly, y,indices, test_size= 0.33, random_state=42)

#Model
log_reg_model = LogisticRegression(penalty= 'l2', C = 5.0)
log_reg_model.fit(Xpoly_train, ytrain)
log_reg_model.intercept_
log_reg_model.coef_

#****************************************************
#Predictions
predictions_log_reg = log_reg_model.predict(Xpoly_test)
df2['preds_log_reg'] = predictions

#Plot logistic regression predictions

def plot_data_preds(df, model_name, preds_col):
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
model_name = 'Logistic Regression model, q = 1, C = 1'
plot_data_preds(df2, model_name, preds_col)


#******************* Part b knn model  *********************************
#i. Choose k
def choose_k_knn(X, y, k_range, plot_color):
    '''knn - Implement 5 fold cross validation for determinine optimal k'''
    
    #Param setup
    kf = KFold(n_splits = 5)
    mean_error=[]; std_error=[];
       
    #Loop through each k fold
    for k in k_range:
        mse_temp = []
        model = KNeighborsClassifier(n_neighbors = k, weights= 'uniform')
                
        for train, test in kf.split(X):
            
            model.fit(X[train], y[train])
            ypred = model.predict(X[test])
            mse = mean_squared_error(y[test],ypred)
            mse_temp.append(mse)
        
        #Get mean & variance
        mean_error.append(np.array(mse_temp).mean())
        std_error.append(np.array(mse_temp).std())
        
    #Plot
    plt.errorbar(k_range, mean_error, yerr=std_error, color = plot_color)
    plt.xlabel('k')
    plt.ylabel('Mean square error')
    plt.title('kNN - 5 fold CV')
    plt.show()

#Implement
k_range = [2,3,5,7,8,10,15,20,25,40, 60, 100]
plot_color = 'orange'
choose_k_knn(X, y, k_range, plot_color)    

#ii. Choose q
def choose_q_knn_cv(X, y, q_range, plot_color):
    '''Implement 5 fold cross validation for testing 
    regression model (lasso or ridge) and plot results'''
    
    #Param setup
    kf = KFold(n_splits = 5)
    mean_error=[]; std_error=[];
    model = KNeighborsClassifier(n_neighbors = 15, weights= 'uniform')
    
    #Loop through each k fold
    for q in q_range:
        mse_temp = []
        Xpoly = PolynomialFeatures(q).fit_transform(X) 
                
        for train, test in kf.split(Xpoly):
            model.fit(Xpoly[train], y[train])
            ypred = model.predict(Xpoly[test])
            mse = mean_squared_error(y[test],ypred)
            mse_temp.append(mse)
        
        #Get mean & variance
        mean_error.append(np.array(mse_temp).mean())
        std_error.append(np.array(mse_temp).std())
        
    #Plot
    plt.errorbar(q_range, mean_error, yerr=std_error, color = plot_color)
    plt.xlabel('q')
    plt.ylabel('Mean square error')
    plt.title('Choice of q in knn - 5 fold CV')
    plt.show()

#q range 
q_range = [1,2,3,4,5, 6,7]
plot_color = 'r'
choose_q_knn_cv(X, y, q_range, plot_color)

#Final
#Model
knn_model = KNeighborsClassifier(n_neighbors = 15, weights= 'uniform')
knn_model.fit(X[indices_train], y[indices_train])
predictions_knn = knn_model.predict(X[indices_test])

#Predictions
df2['preds_knn'] = predictions_knn
preds_col = 'preds_knn'
model_name = 'Logistic Regression model, q = 1, C = 1'
plot_data_preds(df2, model_name, preds_col)

#******************* Part c Confusion matrices  *********************************
#Logistic regression
print(confusion_matrix(ytest, predictions_log_reg))
#knn
print(confusion_matrix(ytest, predictions_knn))

#Baseline model
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X[indices_train], y[indices_train])
predictions_dummy = dummy_clf.predict(X[indices_test])
#Confusion matrix
print(confusion_matrix(ytest, predictions_dummy))

#******************* Part d - ROC Curve *************
def plot_roc_models(Xtest, ytest, log_reg_model, knn_model, dummy_clf):
    'Plot ROC Curve of implemented models'
    
    #Logistic Regression model
    scores = log_reg_model.decision_function(Xtest)
    fpr, tpr, _= roc_curve(ytest, scores)
    plt.plot(fpr,tpr, label = 'Logistic Regression')

    #knn model
    scores = knn_model.predict_proba(Xtest)
    fpr, tpr, _= roc_curve(ytest, scores[:, 1])
    plt.plot(fpr,tpr, color = 'r', label = 'knn')

    #Baseline Model
    scores_bl = dummy_clf.predict_proba(Xtest)
    fpr, tpr, _= roc_curve(ytest, scores_bl[:, 1])
    plt.plot(fpr,tpr, color = 'orange', label = 'baseline model')
    
    #Random Choice
    plt.plot([0, 1], [0, 1],'g--') 

    #Labels
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve') #  - Logistic Regression')

    plt.legend(['Logistic Regression', 'knn', 'Baseline ','Random Classifier'])
    plt.show()    
    
#Implement
plot_roc_models(Xtest, ytest, log_reg_model, knn_model, dummy_clf)

#*********************** Part 2 - Dataset 2  ********************
#The same code was ran again for the second dataset, replacing the dataset week4_d1.txt with week4_d2.txt
data = pd.read_csv('week4_d2.txt',)
