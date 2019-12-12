# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:29:48 2019

@author: Brian
"""
import numpy as np
from sklearn.naive_bayes import MultinomialNB,GaussianNB,ComplementNB
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from kerasClassifier import predictTest as pt

#Main code for testing different algorithms. Commented code are algorithms testing prior. Uncomment in order to test again

def predictTest(trainFeatures,trainLabels,testFeatures,i):
    #model=MultinomialNB(alpha=.01)
    #model= MLPClassifier(random_state=2,hidden_layer_sizes=[100, 100],max_iter=1000)
    #model = SVC(kernel="rbf",probability=True,gamma='scale')
    #model=LinearSVC(penalty="l2")
    #model=RandomForestClassifier(n_estimators=100, max_depth=12,random_state=2)
    ch2=SelectKBest(chi2,k=i)
    train=ch2.fit_transform(trainFeatures, trainLabels)
    test= ch2.transform(testFeatures)
    
    #lsvc = LinearSVC(C=20, penalty="l1", dual=False)
    #clf = Pipeline([('feature_selection', SelectFromModel(lsvc)), ('classification', model)])
    #clf.fit(trainFeatures, trainLabels)
    #trainFeaturesN = select.fit_transform(trainFeatures,trainLabels)
    return pt(train,trainLabels,test,i)
#    model.fit(train,trainLabels)
#    predicted = model.predict_proba(test)[:,1]
#    return predicted
