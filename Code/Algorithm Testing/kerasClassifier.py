   # -*- coding: utf-8 -*-
"""
Demo of 10-fold cross-validation using Gaussian naive Bayes on spam data

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as pl
import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
#seed selection
from numpy.random import seed
seed(2)
import tensorflow as tf
tf.random.set_seed(2)


batch_size = 30
num_classes = 1
epochs = 30

def aucCV(features,labels):
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    cvScores=[]
    for train,test in kfold.split(features,labels):
        model= Sequential()
        model.add(Dense(512, activation='relu', input_dim=30))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        #model.add(Dense(num_classes, activation='softmax'))
        model.add(Dense(1, activation='sigmoid'))
        
        #test_binary = to_categorical(validationLabels)
        #train_binary= to_categorical(labels[test])
        
        model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
        model.fit(features[train], labels[train], batch_size=batch_size, epochs=epochs, verbose=0,use_multiprocessing=True)
        scores=model.evaluate(features[test],labels[test],verbose=0,batch_size=batch_size)
        cvScores.append(scores[1] * 100)
        
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvScores), np.std(cvScores)))
    return cvScores

def predictTest(trainFeatures,trainLabels,testFeatures):
    #trainFeatures,trainLabels,validationFeatures,validationLabels=preproc(trainFeatures,trainLabels)
    keras.backend.clear_session()
    model= Sequential()
    model.add(Dense(512, activation='relu'))
    #model.add(Dense(8, activation='relu', input_shape=(i,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    #model.add(Dense(8, activation='relu'))
    #model.add(Dense(num_classes, activation='softmax'))
    model.add(Dense(1, activation='sigmoid'))
    
    #test_binary = to_categorical(testLabels)
    #train_binary= to_categorical(trainLabels)
    
    model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    model.fit(trainFeatures, trainLabels, batch_size=batch_size, epochs=epochs, verbose=0,validation_split=0.3)
    
    # Use predict_proba() rather than predict() to use probabilities rather
    # than estimated class labels as outputs
    testOutputs = model.predict_proba(testFeatures)#[:,1]
    keras.backend.clear_session()
    del model
    return testOutputs

# splits train data into 75% training data and 25% validation data
def preproc(trainFeatures,trainLabels):
    trainData=np.append(trainFeatures,trainLabels.reshape(-1,1),axis=1)
    x,y=trainData.shape
    train =trainData[:int(x*.75),:] #first 3/4ths
    validation=trainData[int(x*.75):x:1,:] #last 1/4th
    
    trainFeatures = train[:,:-1]
    trainLabels = train[:,-1]
    validFeatures = validation[:,:-1]
    validLabels = validation[:,-1]
    return trainFeatures,trainLabels,validFeatures,validLabels

# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    data = np.loadtxt('spamTrain1.csv',delimiter=',')
    # Randomly shuffle rows of data set then separate labels (last column)
    shuffleIndex = np.arange(np.shape(data)[0])
    np.random.shuffle(shuffleIndex)
    data = data[shuffleIndex,:]
    features = data[:,:-1]
    labels = data[:,-1]
    
    
    
    # Evaluating classifier accuracy using 10-fold cross-validation
    print("10-fold cross-validation mean AUC: ",
          np.mean(aucCV(features,labels)))
    
    x,y=data.shape
    trainData = data[:int(x*.75),:] # first 3/4ths
    testData = data[int(x*.75):x:1,:] # last 1/4ths
    #testData = np.loadtxt(testDataFilename,delimiter=',')
    
    # Randomly shuffle rows of training and test sets then separate labels
    # (last column)
    shuffleIndex = np.arange(np.shape(trainData)[0])
    np.random.shuffle(shuffleIndex)
    trainData = trainData[shuffleIndex,:]
    trainFeatures = trainData[:,:-1]
    trainLabels = trainData[:,-1]
    
    shuffleIndex = np.arange(np.shape(testData)[0])
    np.random.shuffle(shuffleIndex)
    testData = testData[shuffleIndex,:]
    testFeatures = testData[:,:-1]
    testLabels = testData[:,-1]
    
    testOutputs = predictTest(trainFeatures,trainLabels,testFeatures)
    print("Test set AUC: ", roc_auc_score(testLabels,testOutputs))
    
    # Examine outputs compared to labels
    sortIndex = np.argsort(testLabels)
    nTestExamples = testLabels.size
    pl.subplot(2,1,1)
    pl.plot(np.arange(nTestExamples),testLabels[sortIndex],'b.')
    pl.xlabel('Sorted example number')
    pl.ylabel('Target')
    pl.subplot(2,1,2)
    pl.plot(np.arange(nTestExamples),testOutputs[sortIndex],'r.')
    pl.xlabel('Sorted example number')
    pl.ylabel('Output (predicted target)')
    