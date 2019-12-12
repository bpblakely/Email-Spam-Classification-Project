# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 14:39:59 2019

@author: Brian
"""

# -*- coding: utf-8 -*-
"""
Script used to evaluate classifier accuracy

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as pl
from sklearn.metrics import roc_auc_score,roc_curve
from RandomForest import predictTest

nRuns = 10
desiredFPR = 0.01
trainDataFilename = 'spamTrainAll.csv' #spamTrainAll is all the 3000 examples put into 1 file
#testDataFilename = 'spamTrain22.csv'
#testDataFilename = 'spamTest.csv'

def tprAtFPR(labels,outputs,desiredFPR):
    fpr,tpr,thres = roc_curve(labels,outputs)
    # True positive rate for highest false positive rate < 0.01
    maxFprIndex = np.where(fpr<=desiredFPR)[0][-1]
    fprBelow = fpr[maxFprIndex]
    fprAbove = fpr[maxFprIndex+1]
    # Find TPR at exactly desired FPR by linear interpolation
    tprBelow = tpr[maxFprIndex]
    tprAbove = tpr[maxFprIndex+1]
    tprAt = ((tprAbove-tprBelow)/(fprAbove-fprBelow)*(desiredFPR-fprBelow) 
             + tprBelow)
    return tprAt,fpr,tpr

#calculates the TPR and Acc over k many iterations
def mean(allTrainData,k,randomSize):
    x,y=allTrainData.shape
    
    evaluation=np.zeros([k,2])
    tAcc=np.zeros([k,2])
    tTprFpr=np.zeros([k,2])
    #mfpr=np.empty()
    for i in range(0,k):
        shuffleIndex = np.arange(np.shape(allTrainData)[0])
        np.random.shuffle(shuffleIndex)
        trainData= allTrainData[shuffleIndex,:]
        if(randomSize):
            fraction=np.random.uniform(.5,.85)
        else:
            fraction=2/3
        train = trainData[:int(x*fraction),:] # fraction of data
        test= trainData[int(x*fraction):x:1,:] #  (1-fraction) of data
        
        trainFeatures = train[:,:-1]
        trainLabels = train[:,-1]
        testFeatures = test[:,:-1]
        testLabels = test[:,-1]
        testOutputs = predictTest(trainFeatures,trainLabels,testFeatures)
        aucTestRun = roc_auc_score(testLabels,testOutputs)
        tprAtDesiredFPR,fpr,tpr = tprAtFPR(testLabels,testOutputs,desiredFPR)
        evaluation[i,0] = aucTestRun
        evaluation[i,1] = tprAtDesiredFPR
        tAcc[i]=np.sum(evaluation[:,0])/(i+1)
        tTprFpr[i]=np.sum(evaluation[:,1])/(i+1)
        
    mAcc=np.sum(evaluation[:,0])/k
    mTprFpr=np.sum(evaluation[:,1])/k
    
    pl.ion()
    pl.plot(fpr,tpr)

    print(f'All metrics are the mean over {k} iterations')
    if(not randomSize):
        print(f'Data split: {int(fraction*100)} % train | {int((1-fraction)*100)} % test')
    else:
        print(f'Data split: a random distrobution between 50 % to 85 % train : test split')
    print(f'Test set AUC: {mAcc*100} %')
    print(f'Mean TPR at FPR = {desiredFPR}: {mTprFpr*100} %')
    pl.xlabel('False positive rate')
    pl.ylabel('True positive rate')
    pl.title('ROC curve for spam detector')    
    pl.show()
    
# Option to plot the data after every step, useful for small batches
#    y=np.arange(1,k+1,1)
#    pl.ion()
#    pl.figure()
#    pl.xlabel('K (number of iterations)')
#    pl.ylabel('Acc')
#    pl.plot(y,tAcc,'-bo',label='Accuracy')
#    pl.show()
#    pl.figure()
#    pl.plot(y,tTprFpr,'-ro',label='TPR@ 1%FPR') 
#    pl.xlabel('K (number of iterations)')
#    pl.ylabel('TPRFPR')
#    pl.title('Accuracy and TPR @ 1% FPR as iterations increase') 
#    pl.show()
trainData = np.loadtxt(trainDataFilename,delimiter=',')
mean(trainData,100,0)

