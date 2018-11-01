# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 19:55:41 2018

@author: Balaji
"""

import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics, linear_model
from sklearn.model_selection import KFold, train_test_split

#classification
def classification(test,testLabel):
	n = test.shape[0]
	i = 0
	accuracy = {}   
	while(i < n):
		func_x = 0
		ls = list(test[i,:])
		for w, x in zip(est_beta, ls):
			func_x = func_x + (w * x)
		if abs(func_x) > abs(func_x-1):
			y_pred = 1
			if abs(func_x - 1) > abs(func_x - 2):
				y_pred = 2
		else:
			y_pred = 0
			if abs(func_x) > abs(func_x - 2):
				y_pred = 2
		acc = testLabel[i]/y_pred
		accuracy[i] = acc
		i+=1
	avg_acc = np.mean(list(accuracy.values()))
	print('Classification score: ', avg_acc)   


#Data preprocessing
df = np.loadtxt("iris.data", dtype="str", delimiter=",")
rows = df.shape[0]
split_train = (rows*80)//100
train = df[0:split_train, :-1].astype(np.float)
test = df[split_train:, :-1].astype(np.float)
train = np.hstack((train,np.ones((train.shape[0], 1))))
test = np.hstack((test,np.ones((test.shape[0], 1))))
#print(train)
#print(test)
#Assigning integer values from 0 to 2 for data Labels 
trainLabel = df[0:train.shape[0],-1]
trainLabel[trainLabel == "Iris-setosa"] = "0"
trainLabel[trainLabel == "Iris-virginica"] = "1"
trainLabel[trainLabel == "Iris-versicolor"] = "2"
#print(trainLabel)
testLabel = df[train.shape[0]:, -1]
testLabel[testLabel == "Iris-setosa"] = "0"
testLabel[testLabel == "Iris-virginica"] = "1"
testLabel[testLabel == "Iris-versicolor"] = "2"
#print(testLabel)
#Linear regression
#Beta_cap = (A^T.A)^-1*Y
est_beta = None
matA = train
trans_matA = np.transpose(matA)
#print(trans_matA)
matY = trainLabel.astype(np.float)
testLabel = testLabel.astype(np.float)
t1 = np.dot(trans_matA, matA)
t1 = np.linalg.inv(t1)
#print(t1)
t2 = np.dot(t1, trans_matA)
#print(t2)
est_beta = np.dot(t2,matY)
print(est_beta)


for i in range(2,10):
    kf = KFold(n_splits=i)
    s = 0
    #train,test, trainLabel, testLabel = train_test_split(df, trainLabel, test_size=0.2)
    lm = linear_model.LinearRegression()
    model = lm.fit(train, trainLabel)
    predictions = lm.predict(test)
    matY = matY.reshape(-1,1)
    predictions = cross_val_predict(lm, matY, trainLabel, cv=i)
    accuracy = metrics.r2_score(trainLabel.astype(np.float64), predictions)
    print(i,'-fold Cross-validation scores:', accuracy)
'''for train_index, test_index in kf.split(rows):
        train1, test1 = train[train_index], test[test_index]
        trainLabel1, testLabel1 = trainLabel[train_index], testLabel[test_index]
        classifier = classification(train,trainLabel)
    
#scores = cross_val_score(lm, matY, trainLabel, cv=10)
'''
    

'''from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(test, testLabel) 
print(clf.predict(test))
print(clf.score(test, testLabel))'''

#from sklearn.neighbors import KNeighborsClassifier
#neigh = KNeighborsClassifier(n_neighbors=5)
#neigh.fit(test, testLabel) 
#print(neigh.predict(test))
#print(neigh.score(test, testLabel))




