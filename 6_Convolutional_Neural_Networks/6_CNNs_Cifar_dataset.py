# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 22:06:37 2020

@author: Hannah Craddock
"""
from PIL import Image
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True
import sys
from sklearn.dummy import DummyClassifier
import time
from sklearn.metrics import accuracy_score

%matplotlib qt
plt.rc('font', size=15);

# Model / data parameters
num_classes = 10
input_shape = (32, 32, 3)

#********************************************
#Part 1.

#Part a - Convolve Function
def convolve(matrix, kernel):
    ''' Convolve a nxn matrix with a kxk kernel      '''

    #Get shapes
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = matrix.shape[0]
    yImgShape = matrix.shape[0]

    # Shape of Output Convolution
    xOutput = int((xImgShape - xKernShape + 2) + 1)
    yOutput = int((yImgShape - yKernShape + 2) + 1)
    output = np.zeros((xOutput, yOutput))

    # Iterate through image
    for y in range(matrix.shape[1]):
        # Exit Convolution
        if y > matrix.shape[1] - yKernShape:
            break
        for x in range(matrix.shape[0]):
            # Go to next row once kernel is out of bounds
            if x > matrix.shape[0] - xKernShape:
                break
            output[x, y] = (kernel * matrix[x: x + xKernShape, y: y + yKernShape]).sum()

    return output

#Apply Function
matrix = np.random.randint(1,8, size = (6,6))
kernel = np.random.randint(1,8, size = (3,3))
conv_output = convolve(matrix, kernel)


#*************************
#Part b 
im = Image.open('sunflower.jpg')
im = im.resize((200,200))
rgb = np.array(im.convert('RGB'))
r = rgb[:,:,0] # array of R pixels
#Image.fromarray(np.uint(r)).show()


#Apply function w/ kernel1
kernel1 = np.array([[-1,-1,1],[-1,8,-1],[-1,-1,-1]])
conv_output1 = convolve(r, kernel1)
conv_output1
Image.fromarray(np.uint(conv_output1)).show()


#Apply function w/ kernel2
kernel2 = np.array([[0,-1,0],[-1,8,-1],[0,-1,0]])
conv_output2 = convolve(r, kernel2)
#Display image
Image.fromarray(np.uint(conv_output2)).show()


#****************************************************************************************************
#Part 2

#Data - split between train and test sets
(x_train_orig, y_train_orig), (x_test, y_test) = keras.datasets.cifar10.load_data()

#Split data
n=5000
x_train = x_train_orig[1:n]; y_train = y_train_orig[1:n]
#x_test=x_test[1:500]; y_test=y_test[1:500]

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
print("orig x_train shape:", x_train.shape)

#convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#Train model
use_saved_model = False
if use_saved_model:
    model = keras.models.load_model("cifar.model")
else:
    model = keras.Sequential()
    model.add(Conv2D(16, (3,3), padding='same', input_shape = x_train.shape[1:], activation='relu'))
    model.add(Conv2D(16, (3,3), strides=(2,2), padding='same', activation='relu'))
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3,3), strides=(2,2), padding='same', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l1(0.0001)))
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    model.summary()
    
    #Training
    #Training params
    batch_size = 128
    epochs = 20
    
    #Time it
    start = time.time()
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    end = time.time()
    print(end - start)
    #Save model 
    model.save("cifar.model")
    
    #Plots
    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss'); plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

#Prediction 
#Training set 
preds = model.predict(x_train)
y_pred = np.argmax(preds, axis=1) #Highest prob
y_train1 = np.argmax(y_train, axis=1)
print(classification_report(y_train1, y_pred))
print(confusion_matrix(y_train1,y_pred))
train_accuracy = accuracy_score(y_train1, y_pred)

#Test set 
y_preds_test = model.predict(x_test)
y_preds_test = np.argmax(y_preds_test, axis=1)
y_test1 = np.argmax(y_test, axis=1)
print(classification_report(y_test1, y_preds_test))
print(confusion_matrix(y_test1,y_preds_test))
test_accuracy = accuracy_score(y_test1, y_preds_test)

#Accuracy 
#Model prediction results 
y_preds_test = model.predict(x_test)
y_preds_test = np.argmax(y_preds_test, axis=1) #Highest prob
#Format 
y_test1 = np.argmax(y_test, axis=1)
test_accuracy = accuracy_score(y_test1, y_preds_test)

#**********************************************
#Part b.ii. Baseline Model
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(x_train, y_train)

#Predict on training set
predictions_dummy = dummy_clf.predict(x_train)
y_pred_dummy = np.argmax(predictions_dummy, axis=1)

print(classification_report(y_train1, y_pred_dummy))
print(confusion_matrix(y_train1,y_pred_dummy))

#Baseline v2
y_pred_bl = 8*np.ones(len(y_train1))

print(classification_report(y_train1, y_pred_bl))
print(confusion_matrix(y_train1,y_pred_bl))

#Test set
#Baseline v2
y_pred_bl = 8*np.ones(len(y_test1))

print(classification_report(y_test1, y_pred_bl))
print(confusion_matrix(y_test1, y_pred_bl))

#*********************************************************
#Part iii. Timing

def model_train_sizes(x_train_orig, y_train_orig, x_test, y_test, train_size, batch_size, epochs):
    
    #Figure
    fig, axs = plt.subplots(4, 2)
    results = []
    count = 0
    
    for n in train_size:
        'Perform training for each size of n'
        #Split data
        x_train = x_train_orig[1:n]; y_train = y_train_orig[1:n]
        
        #Format data
        x_train = x_train.astype("float32") / 255
        #Convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)   
        
        #Train model
        start = time.time()
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        end = time.time()
        timeX = (end - start)
        print('time train = {}'.format(timeX))
        
        #Model prediction results 
        y_preds_train = model.predict(x_train)
        y_preds_train = np.argmax(y_preds_train, axis=1) #Highest prob
        #Format 
        y_train1 = np.argmax(y_train, axis=1)
        train_accuracy = accuracy_score(y_train1, y_preds_train)
        print('train accuracy = {}'.format(train_accuracy))
        
        #Model prediction results 
        y_preds_test = model.predict(x_test)
        y_preds_test = np.argmax(y_preds_test, axis=1) #Highest prob
        #Format 
        y_test1 = np.argmax(y_test, axis=1)
        test_accuracy = accuracy_score(y_test1, y_preds_test)
        print('test accuracy = {}'.format(test_accuracy))
        
        #Results
        d = {
            'training size' : n,
        'train time': timeX,  
        'train accuracy' :  train_accuracy,
        'test accuracy' :  test_accuracy,
        }
        results.append(d)
        
        #Plots
        #Accuracy
        axs[count, 0].plot(history.history['accuracy'])
        axs[count, 0].plot(history.history['val_accuracy'])
        axs[count, 0].set_title('Model Accuracy, train size = {}'.format(n))
        axs[count, 0].set(xlabel='epoch', ylabel='accuracy')
        axs[count, 0].legend(['train', 'val'], loc='upper left')
        #Loss
        axs[count, 1].plot(history.history['loss'])
        axs[count, 1].plot(history.history['val_loss'])
        axs[count, 1].set_title('Model Loss, train size = {}'.format(n))
        axs[count, 1].set(xlabel='epoch', ylabel='loss')
        axs[count, 1].legend(['train', 'val'], loc='upper left')
        #Update count
        count = count + 1
        
    #Show plot
    plt.show()
    #Dataframe of results
    df_results = pd.DataFrame(results)
    
    return df_results 

#Apply
train_size = [5000, 10000, 20000, 40000]  
df_results3 = model_train_sizes(x_train_orig, y_train_orig, x_test, y_test, train_size, batch_size, epochs)


#***********************************************
#Part 4. Inspect varying L1 reguliser

def model_vary_L1(x_train, y_train, x_test, y_test, list_l1):
    'Train model for varying L1'
    #Figure
    fig, axs = plt.subplots(8, 2)
    count = 0 
    results = []
    
    for l1X in list_l1:
        model = keras.Sequential()
        model.add(Conv2D(16, (3,3), padding='same', input_shape = x_train.shape[1:], activation='relu'))
        model.add(Conv2D(16, (3,3), strides=(2,2), padding='same', activation='relu'))
        model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
        model.add(Conv2D(32, (3,3), strides=(2,2), padding='same', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l1(l1X)))
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
        #model.summary()
    
        #Training  
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        
        #Model prediction results 
        y_preds_train = model.predict(x_train)
        y_preds_train = np.argmax(y_preds_train, axis=1) #Highest prob
        #Format 
        y_train1 = np.argmax(y_train, axis=1)
        train_accuracy = accuracy_score(y_train1, y_preds_train)
        
        #Model prediction results 
        y_preds_test = model.predict(x_test)
        y_preds_test = np.argmax(y_preds_test, axis=1) #Highest prob
        #Format 
        y_test1 = np.argmax(y_test, axis=1)
        test_accuracy = accuracy_score(y_test1, y_preds_test)
        
        #Results
        d = {
            'L1 weight' : l1X, 
        'train accuracy' :  train_accuracy,
        'test accuracy' :  test_accuracy,
        
        }
        results.append(d)
    
        #Plots
        axs[count, 0].plot(history.history['accuracy'])
        axs[count, 0].plot(history.history['val_accuracy'])
        axs[count, 0].set_title('Model Accuracy, L1 weight = {}'.format(l1X))
        axs[count, 0].set(xlabel='epoch', ylabel='accuracy')
        axs[count, 0].legend(['train', 'val'], loc='upper left')
        #Loss
        axs[count, 1].plot(history.history['loss'])
        axs[count, 1].plot(history.history['val_loss'])
        axs[count, 1].set_title('Model Loss, L1 weight size = {}'.format(l1X))
        axs[count, 1].set(xlabel='epoch', ylabel='loss')
        axs[count, 1].legend(['train', 'val'], loc='upper left')
        #Update count
        count = count + 1
        
    #Dataframe of results
    df_results = pd.DataFrame(results)
    #Show plot
    plt.show()
    
    return df_results 

#Apply function
list_l1 = [0.0001, 0.01, 0.1, 1, 10, 50, 100, 1000]
df_results2 = model_vary_L1(x_train, y_train, x_test, y_test, list_l1) 

#****************************************************************************
#Part c - Model
    
#Train model
use_saved_model = False
if use_saved_model:
    model2 = keras.models.load_model("cifar.model")
else:
    model2 = keras.Sequential()
    model2.add(Conv2D(16, (3,3), padding='same', input_shape = x_train.shape[1:], activation='relu'))
    model2.add(Conv2D(16, (3,3), padding='same', activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model2.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Dropout(0.5))
    model2.add(Flatten())
    model2.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l1(0.0001)))
    model2.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    model2.summary()
    
    #Training
    #Training params
    batch_size = 128
    epochs = 20
    
    #Time it
    start = time.time()
    history = model2.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    end = time.time()
    print(end - start)
    #Save model 
    model2.save("cifar2.model")
    
#Performance
#Plots
fig, axs = plt.subplots(2, 1)
#Accuracy
axs[0].plot(history.history['accuracy'])
axs[0].plot(history.history['val_accuracy'])
axs[0].set_title('Model Accuracy, train size = {}'.format(n))
axs[0].set(xlabel='epoch', ylabel='accuracy')
axs[0].legend(['train', 'val'], loc='upper left')
#Loss
axs[1].plot(history.history['loss'])
axs[1].plot(history.history['val_loss'])
axs[1].set_title('Model Loss, train size = {}'.format(n))
axs[1].set(xlabel='epoch', ylabel='loss')
axs[1].legend(['train', 'val'], loc='upper left')

#***************************
#Prediction 
#Training set 
preds = model2.predict(x_train)
y_preds_train = np.argmax(preds, axis=1) #Highest prob
y_train1 = np.argmax(y_train, axis=1)
print(classification_report(y_train1, y_pred))
print(confusion_matrix(y_train1,y_pred))
train_accuracy = accuracy_score(y_train1, y_preds_train)

#Test set 
y_preds_test = model2.predict(x_test)
y_preds_test = np.argmax(y_preds_test, axis=1)
y_test1 = np.argmax(y_test, axis=1)
print(classification_report(y_test1, y_preds_test))
print(confusion_matrix(y_test1,y_preds_test))
test_accuracy = accuracy_score(y_test1, y_preds_test)
test_accuracy

#**********************************************************************************************
#Model Part d

model3 = keras.Sequential()
model3.add(Conv2D(8, (3, 3), padding= 'same', input_shape= x_train.shape[1: ], activation = 'relu' ) )
model3.add(Conv2D(8, (3, 3), strides = (2,2) , padding= 'same', activation = 'relu' ) )
model3.add(Conv2D(16, (3, 3), padding= 'same',  activation = 'relu') )
model3.add(Conv2D(16, (3, 3), strides = (2, 2) , padding= 'same', activation = 'relu' ) )
model3.add(Conv2D(32, (3, 3), padding= 'same', activation = 'relu') )
model3.add(Conv2D(32, (3, 3), strides = (2,2) , padding= 'same', activation = 'relu'))
model3.add(Dropout (0.5))
model3.add(Flatten( ))
model3.add(Dense(num_classes, activation = 'softmax', kernel_regularizer = regularizers.l1(0.0001)))
model3.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
model3.summary()

#Training
#Training params
batch_size = 128
epochs = 20

#Time it
start = time.time()
history3 = model3.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
end = time.time()
print(end - start)
#Save model 
model.save("cifar.model")

#**************
# Effect on train size

def model_train_sizesII(x_train_orig, y_train_orig, x_test, y_test, train_size, batch_size, epochs):
    
    #Figure
    fig, axs = plt.subplots(4, 2)
    results = []
    count = 0
    
    for n in train_size:
        'Perform training for each size of n'
        #Split data
        x_train = x_train_orig[1:n]; y_train = y_train_orig[1:n]
        
        #Format data
        x_train = x_train.astype("float32") / 255
        #Convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)   
        
        #Train model
        start = time.time()
        history = model3.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        end = time.time()
        timeX = (end - start)
        print('time train = {}'.format(timeX))
        
        #Model prediction results 
        y_preds_train = model.predict(x_train)
        y_preds_train = np.argmax(y_preds_train, axis=1) #Highest prob
        #Format 
        y_train1 = np.argmax(y_train, axis=1)
        train_accuracy = accuracy_score(y_train1, y_preds_train)
        print('train accuracy = {}'.format(train_accuracy))
        
        #Model prediction results 
        y_preds_test = model.predict(x_test)
        y_preds_test = np.argmax(y_preds_test, axis=1) #Highest prob
        #Format 
        y_test1 = np.argmax(y_test, axis=1)
        test_accuracy = accuracy_score(y_test1, y_preds_test)
        print('test accuracy = {}'.format(test_accuracy))
        
        #Results
        d = {
            'training size' : n,
        'train time': timeX,  
        'train accuracy' :  train_accuracy,
        'test accuracy' :  test_accuracy,
        }
        results.append(d)
        
        #Plots
        #Accuracy
        axs[count, 0].plot(history.history['accuracy'])
        axs[count, 0].plot(history.history['val_accuracy'])
        axs[count, 0].set_title('Model Accuracy, train size = {}'.format(n))
        axs[count, 0].set(xlabel='epoch', ylabel='accuracy')
        axs[count, 0].legend(['train', 'val'], loc='upper left')
        #Loss
        axs[count, 1].plot(history.history['loss'])
        axs[count, 1].plot(history.history['val_loss'])
        axs[count, 1].set_title('Model Loss, train size = {}'.format(n))
        axs[count, 1].set(xlabel='epoch', ylabel='loss')
        axs[count, 1].legend(['train', 'val'], loc='upper left')
        #Update count
        count = count + 1
        
    #Show plot
    plt.show()
    #Dataframe of results
    df_results = pd.DataFrame(results)
    
    return df_results 

#Apply
train_size = [5000, 10000, 20000, 40000]  
df_resultsII = model_train_sizesII(x_train_orig, y_train_orig, x_test, y_test, train_size, batch_size, epochs)
