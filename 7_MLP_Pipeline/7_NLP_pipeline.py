# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 22:12:19 2020

@author: Hannah Craddock
"""

#*******************************

#Data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import json_lines
import nltk
from google_trans_new import google_translator #from googletrans import Translator
nltk.download('stopwords')
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_curve, auc
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from google_trans_new import google_translator  

#Plot
%matplotlib qt
SMALL_SIZE = 8
MEDIUM_SIZE = 28
BIGGER_SIZE = 30

plt.rc('font', size=MEDIUM_SIZE);
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


#*************************
#Data
X_orig =[] ; y=[] ; z=[]
x_label = 'text'
y_label ='voted_up'
z_label = 'early_access' 

with open ( 'reviews_228.jl' , 'rb') as f :
    for item in json_lines.reader(f):
        X_orig.append(item[x_label])
        y.append(item[y_label])
        z.append(item[z_label]) 
 

#Data preprocessing - Translate data to english
translator = google_translator()   #print(translator.translate('안녕하세요.'))

X_df = pd.DataFrame()
X_df['text'] = X_orig
X_df['translated'] = X_orig

#Translate column
X_df['translated'] = X_df['text'].apply(translator.translate, lang_tgt='en') #.apply(getattr, args=('text',))

#values
X = list(df['translated'].values)

#**************************************************
#Model 1 - SVM

#ii. Use k fold cross validation to train model
def SVM_choose_c(X, y, c_range, num_splits, y_label, plot_color):
    '''Support Vector Classification - Implement 5 fold cross validation to determinine optimal C'''
    
    #Param setup
    kf = KFold(n_splits = num_splits, shuffle = True)
    mean_f1 =[]; std_f1 =[];
    X = np.array(X)
    y = np.array(y)
       
    #Loop through each k fold
    for c_param in c_range:
        print('c = {}'.format(c_param))
        f1_temp = []
        
        model = Pipeline([('vect', CountVectorizer(stop_words = nltk.corpus.stopwords.words('english'))),
                          ('tfidf', TfidfTransformer()),
                          ('clf', LinearSVC(C = c_param))])
        
        for train_index, test_index in kf.split(X):
            
            model.fit(X[train_index], y[train_index])
            ypred = model.predict(X[test_index])
            f1_temp.append(f1_score(y[test_index],ypred))
        
        #Get mean & variance
        mean_f1.append(np.array(f1_temp).mean())
        std_f1.append(np.array(f1_temp).std())
     
    #Plot
    #plt.figure()
    plt.errorbar(c_range, mean_f1, yerr=std_f1, color = plot_color)
    plt.xlabel('c penalty term')
    plt.ylabel('f1 score')
    plt.title('SVC - CV for penalty term c, X = review text, y = {}'.format(y_label))
    plt.savefig('./svm_{}'.format(y_label))
    plt.show()


#**********************************
#Model 2 - Knn
    
def knn_choose_k(X, y, k_range, num_splits, y_label, plot_color):
    '''knn - Implement 5 fold cross validation for determinine optimal k'''
    
    #Param setup
    kf = KFold(n_splits = num_splits, shuffle = True)
    mean_f1 =[]; std_f1 =[];
    X = np.array(X)
    y = np.array(y)
       
    #Loop through each k fold
    for k in k_range:
        print('k = {}'.format(k))
        f1_temp = []
        
        model = Pipeline([('vect', CountVectorizer(stop_words = nltk.corpus.stopwords.words('english'))),
                          ('tfidf', TfidfTransformer()),
                          ('clf', KNeighborsClassifier(n_neighbors = k, weights= 'uniform'))])
        
        for train_index, test_index in kf.split(X):
            
            model.fit(X[train_index], y[train_index])
            ypred = model.predict(X[test_index])
            f1_temp.append(f1_score(y[test_index],ypred))
        
        
        #Get mean & variance
        mean_f1.append(np.array(f1_temp).mean())
        std_f1.append(np.array(f1_temp).std())
    
    #Plot
    plt.figure(figsize = (12, 10))
    plt.errorbar(k_range, mean_f1, yerr=std_f1, color = plot_color)
    plt.xlabel('k')
    plt.ylabel('f1 score')
    plt.title('CV for k in knn, X = review text, y = {}'.format(y_label))
    plt.savefig('./knn_{}'.format(y_label))
    plt.show()
 

def optimal_models_performance (X, y, optimal_k, optimal_C, y_label):    
    ''' Grid search for optimal nlp classifier models (svm and knn). Plot ROC curves, generate confusion matrices and classification report '''
    
    #1. Split the data
    testSizeX = 0.33 #67:33 split
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size= testSizeX, random_state=42) 
    
    #SVM
    svm_model = Pipeline([('vect', CountVectorizer(stop_words = nltk.corpus.stopwords.words('english'))),
                      ('tfidf', TfidfTransformer()),
                      ('clf', LinearSVC())])
    
    #Knn
    knn_model = Pipeline([('vect', CountVectorizer(stop_words = nltk.corpus.stopwords.words('english'))),
                      ('tfidf', TfidfTransformer()),
                      ('clf', KNeighborsClassifier(n_neighbors = optimal_k, weights= 'uniform'))])
    
    #Dummy classifier
    dummy_model = DummyClassifier(strategy='most_frequent').fit(Xtrain, ytrain)
    
    
    #Grid search
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False)} 
    
    #************************************************
    #Svm: Train svm model
    svm_gs = GridSearchCV(svm_model, parameters, n_jobs=-1)

    #Performance - best performing
    print('*********************************************')
    print('====== \n Results for svm grid search model:')
    
    svm_gs = svm_gs.fit(Xtrain, ytrain)  
    print(svm_gs.best_params_)       
    predicted = svm_gs.predict(Xtest)
    
    print(confusion_matrix(ytest, predicted))
    print(classification_report(ytest, predicted))
    
    #************************************************
    #Train knn model
    knn_gs = GridSearchCV(knn_model, parameters, n_jobs=-1)

    #Performance - best performing 
    print('*********************************************')    
    print('====== \n Results for knn grid search model:')
    
    knn_gs = knn_gs.fit(Xtrain, ytrain) 
    print(knn_gs.best_params_) 
    predicted = knn_gs.predict(Xtest)
    
    print(confusion_matrix(ytest, predicted))
    print(classification_report(ytest, predicted))
    
    #**********************************************
    #Dummy model
    print('*********************************************') 
    print('====== \n Results for dummy model:')
       
    dummy_model_fitted = dummy_model.fit(Xtrain, ytrain)  
    predicted = dummy_model_fitted.predict(Xtest)
    
    print(confusion_matrix(ytest, predicted))
    print(classification_report(ytest, predicted))
    
    #**********************************************
    #ROC plots
    plt.figure()
    
    #svm model 
    scores = svm_gs.decision_function(Xtest)
    fpr, tpr, _= roc_curve(ytest, scores)
    plt.plot(fpr,tpr, label = 'SVM')
    print('SVM AUC = {}'.format(auc(fpr, tpr)))

    #knn model
    scores = knn_gs.predict_proba(Xtest)[:,1]
    fpr, tpr, _= roc_curve(ytest, scores)
    plt.plot(fpr,tpr, color = 'r', label = 'knn')
    print('knn AUC = {}'.format(auc(fpr, tpr)))

    #Baseline Model
    scores_bl = dummy_model_fitted.predict_proba(Xtest)
    fpr, tpr, _= roc_curve(ytest, scores_bl[:, 1])
    plt.plot(fpr,tpr, color = 'orange', label = 'baseline model')
    print('AUC = {}'.format(auc(fpr, tpr)))
    
    #Random Choice
    plt.plot([0, 1], [0, 1],'g--') 

    #Labels
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve. X = review text, y = {}'.format(y_label))

    plt.legend(['Svm', 'Knn', 'Baseline (most freq)','Random Classifier']) 
    plt.savefig('./roc_{}'.format(y_label))
    plt.show()
   
    
#*********************************************************
#Apply Functions

#**************************
#Part 1 X = review text, y = voted up 

#Split x and y ('z') into train and test sets for subsequent model fitting
testSizeX = 0.33 #67:33 split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size= testSizeX, random_state=42)  

#i. SVM - choose C
c_range = np.geomspace(0.001, 1000, num = 7)
c_range = np.concatenate((c_range, c_range/2))
c_range = np.sort(c_range)
#c_range = np.linspace(start=0.001, stop=1000, num=15) 

num_splits = 5
plot_color = 'blue'
SVM_choose_c(X, y, c_range, num_splits, y_label, plot_color)

#Repeat (smaller range)
c_range = np.geomspace(0.001, 10, num = 7)
c_range = np.concatenate((c_range, c_range/2))
c_range = np.sort(c_range)
 
SVM_choose_c(X, y, c_range, num_splits, y_label, plot_color)

#ii. knn - Choose k
k_range = [2,5,10,20, 50, 75, 100, 150, 200, 250, 300]
plot_color = 'green'
knn_choose_k(X, y, k_range, num_splits, y_label, plot_color) 

#iii.Grid search - optimal model
optimal_k = 150
optimal_c = 1
optimal_models_performance (X, y, optimal_k, optimal_c, y_label)

#**************************#**************************#********************************************************************************************************
#Part 2 X = review text, z = early access 

#Split x and y ('z') into train and test sets for subsequent model fitting
testSizeX = 0.33 #67:33 split
Xtrain, Xtest, ztrain, ztest = train_test_split(X, z, test_size= testSizeX, random_state=42) 

#i. SVM - choose C
c_range = np.geomspace(0.001, 1000, num = 10)
c_range = np.concatenate((c_range, c_range/2))
c_range = np.sort(c_range)

num_splits = 5
plot_color = 'orange'
SVM_choose_c(X, z, c_range, num_splits, z_label, plot_color)


#ii. knn - Choose k
k_range = [2,5,10,20, 50, 75, 100, 150, 200, 250, 300]
plot_color = 'green'
knn_choose_k(X, z, k_range, num_splits, z_label, plot_color) 

#Repeat
k_range = [2,3,4,6,8,10,12,15]
knn_choose_k(X, z, k_range, num_splits, z_label, plot_color) 

#iii.Grid search - optimal model
optimal_k = 150
optimal_c = 50
optimal_models_performance(Xtrain, ytrain, Xtest, ytest, optimal_k, optimal_c, z_label)#
