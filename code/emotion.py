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


data = pd.read_csv('ItsPersonal/data/mbti_2_1_emotion.csv')
cleanedData = data.values

numInstances = len(data.values)
numFeatures = len(data.values[0])

# first column is label, second column is text, IGNORE those 2 columns
X = cleanedData[:, 2:numFeatures].tolist()
# y = np.reshape(cleanedData[:, 0], (numInstances, 1))
y = np.reshape(cleanedData[:, 0], (numInstances, 1)).astype(int).flatten()
print(y)



# model = LogisticRegression(penalty='l1')
# model = LogisticRegression(penalty='l2')

# kernels: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
# model = svm.SVC(kernel='linear')
# model = svm.SVC(kernel='precomputed')
# model = svm.SVC(kernel='rbf')
# model = svm.SVC(kernel='sigmoid')
# model = svm.SVC(kernel='precomputed')

# model = RandomForestClassifier(n_estimators=200, max_depth=i, random_state=0)
model = RandomForestClassifier()


# DON'T NEED TO FIT MODEL, CROSS VAL DOES IT FOR YOU

#model.fit(X,y)



# PREDICTING PORTION

# testData = cleanTestData('challengeTestUnlabeled.csv')
# testCleanedData = testData.values

# testNumInstances = len(testData.values)
# testNumFeatures = len(testData.values[0])

# testX = testCleanedData[:, 0:testNumFeatures]
# testPredict = model.predict(testX)



from sklearn.model_selection import cross_val_score
# for i in range(10,15):
    #clf = RandomForestClassifier(n_estimators=200, max_depth=i, random_state=0)
scores = cross_val_score(model, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))




