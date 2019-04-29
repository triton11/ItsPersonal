# Takes in emotion features from IBM Watson and Myers-Briggs as data
# create classifiers such as Random Forest, SVM, Logistic Regression, and Neural Net


import pandas as pd
import numpy as np
from operator import add, sub

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import svm


data = pd.read_csv('ItsPersonal/data/train_data.csv', encoding="ISO-8859-1", engine='python')
cleanedData = data.values

numInstances = len(data.values)
numFeatures = len(data.values[0])

# first column is label, second column is text, IGNORE those 2 columns
X = cleanedData[:, 2:numFeatures].tolist()
# y = np.reshape(cleanedData[:, 0], (numInstances, 1))
y = np.reshape(cleanedData[:, 0], (numInstances, 1)).astype(int).flatten()
print(y)



model = LogisticRegression(penalty='l1')
# model = LogisticRegression(penalty='l2')

# kernels: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
# model = svm.SVC(kernel='linear')
# model = svm.SVC(kernel='precomputed')
# model = svm.SVC(kernel='rbf')
# model = svm.SVC(kernel='sigmoid')
# model = svm.SVC(kernel='precomputed')

# model = RandomForestClassifier(n_estimators=200, max_depth=i, random_state=0)
# model = RandomForestClassifier()








####################################################################
# FOR ACCURACY, DON'T NEED TO FIT MODEL, CROSS VAL DOES IT FOR YOU
####################################################################

from sklearn.model_selection import cross_val_score
import sklearn.metrics # import accuracy_score, precision_score, recall_score

# for i in range(10,15):
    #clf = RandomForestClassifier(n_estimators=200, max_depth=i, random_state=0)
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


######################
# PREDICTING PORTION
######################

model.fit(X,y)

testData = pd.read_csv('ItsPersonal/data/test_data.csv', encoding="ISO-8859-1", engine='python')
testCleanedData = testData.values

numTestInstances = len(testData.values)
numTestFeatures = len(testData.values[0])

testX = testCleanedData[:, 2:numTestFeatures].tolist()
testYActual = np.reshape(testCleanedData[:, 0], (numTestInstances, 1)).astype(int).flatten()
testYPredict = model.predict(testX)


##########################################################################################
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
##########################################################################################

accuracyScore = sklearn.metrics.accuracy_score(testYActual, testYPredict)
print("Accuracy: " + str(accuracyScore.mean()))

precisionScore = sklearn.metrics.precision_score(testYActual, testYPredict, average='weighted')
print("Precision: " + str(precisionScore.mean()))
# print("Precision: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

recallScore = sklearn.metrics.recall_score(testYActual, testYPredict, average='weighted')
print("Recall: " + str(recallScore.mean()))



