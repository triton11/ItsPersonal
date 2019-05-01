#Final
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


data = pd.read_csv('../data/lda_results_train.csv', encoding="ISO-8859-1", engine='python')
cleanedData = data.values

numInstances = len(data.values)
numFeatures = len(data.values[0])
print(numFeatures)

# first column is label, second column is text, IGNORE those 2 columns
X = cleanedData[:, 2:numFeatures].tolist()
# y = np.reshape(cleanedData[:, 0], (numInstances, 1))
y = np.reshape(cleanedData[:, 0], (numInstances, 1)).astype(int).flatten()
print(X[0:5])
print(y[0:5])

model = LogisticRegression(penalty='l1',tol=0.0005,solver='liblinear')
#
from sklearn.model_selection import cross_validate
import sklearn.metrics # import accuracy_score, precision_score, recall_score

#model = svm.SVC(kernel='linear')
#model = svm.SVC(kernel='precomputed')
# model = svm.SVC(kernel='rbf')
# model = svm.SVC(kernel='sigmoid')
# model = svm.SVC(kernel='precomputed')

#model = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=0)
scores = cross_validate(model, X, y, cv=5, scoring='accuracy')
print(scores['train_score'].mean())
print(scores['test_score'].mean())
model.fit(X, y)
testYPredict = model.predict(X)
print(testYPredict[0:10])

	#print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
