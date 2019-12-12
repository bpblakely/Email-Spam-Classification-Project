# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:09:43 2019

Final code used for email spam classifcation

@author: Brian Blakely
"""

from sklearn.ensemble import RandomForestClassifier
import numpy as np
from numpy.random import seed
seed(2)


def predictTest(trainFeatures,trainLabels,testFeatures):
    clf = RandomForestClassifier(n_estimators=400, max_depth=13,random_state=1,min_samples_leaf=1)
    trainFeatures,trainLabels= preproc(trainFeatures,trainLabels)
    clf.fit(trainFeatures,trainLabels)
    testOutputs= clf.predict_proba(testFeatures)
    return testOutputs[:,1]

#takes training data and forcefully changes the labels of all training examples where the features are all 0, but the label is 1 to 0
#returns trainFeatures,trainLabels
def preproc (trainFeatures,trainLabels):
    trainData=np.c_[trainFeatures,trainLabels]
    for i in range(0,trainData.shape[0]):
        if (np.all(trainData[i,:-1]==0) and trainData[i,-1]==1):
            trainData[i,-1]=0
            
    return trainData[:,:-1],trainData[:,-1] 
