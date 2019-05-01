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
trainX = cleanedData[:, 2:numFeatures].tolist()
#print("X " + str(trainX))
trainY = np.reshape(cleanedData[:, 0], (numInstances, 1))
# y = np.reshape(cleanedData[:, 0], (numInstances, 1)).astype(int).flatten()
print(trainY)


##################################
#
# FIVE FOLD CROSS VALIDATION BELOW
#
##################################

def fiveFold(model, cleanedData):

    # shuffle the instances
    np.random.shuffle(cleanedData)

    numInstances = len(cleanedData)
    numFeatures = len(cleanedData[0]) - 1  # last column is label
    #print("numFeatures " + str(numFeatures))
    foldCount = int(numInstances/5)

    X = cleanedData[:, 0:numFeatures]
    y = np.reshape(cleanedData[:, numFeatures], (numInstances, 1))


    xFold1 = X[0:foldCount, :]
    xFold2 = X[foldCount:foldCount*2, :]
    xFold3 = X[foldCount*2: foldCount*3, :]
    xFold4 = X[foldCount*3: foldCount*4, :]
    xFold5 = X[foldCount*4: numInstances, :]
    allXFolds = [xFold1, xFold2, xFold3, xFold4, xFold5]

    yFold1 = y[0:foldCount, :]
    yFold2 = y[foldCount:foldCount*2, :]
    yFold3 = y[foldCount*2: foldCount*3, :]
    yFold4 = y[foldCount*3: foldCount*4, :]
    yFold5 = y[foldCount*4: numInstances, :]
    allYFolds = [yFold1, yFold2, yFold3, yFold4, yFold5]

    collectedTestList = list()
    collectedTrainList = list()

    testAccuracyList = list()
    testPrecisionList = list()
    testRecallList = list()
    trainAccuracyList = list()
    trainPrecisionList = list()
    trainRecallList = list()


    for i in range(5):
        
        testX = allXFolds[i]
        testY = allYFolds[i]
        trainX = None
        trainY = None

        for j in range(5):
            if j != i:
                if trainX is None and trainY is None:
                    trainX = allXFolds[j]
                    trainY = allYFolds[j]
                else:
                    trainX = np.vstack((trainX, allXFolds[j]))
                    trainY = np.vstack((trainY, allYFolds[j]))

        # train model
        model.fit(trainX,trainY.astype(int).flatten())

        # predict on fold data
        predictedYTest = model.predict(testX).reshape((testY.size, 1))
        #predictedYTest = np.array([1 if x > 0.5 else 0 for x in predictedYTest]).reshape((testY.size, 1))
        predictVsTest = (predictedYTest == testY)
        # predict on original training data
        predictedYTrain = model.predict(trainX).reshape((trainY.size, 1))
        #predictedYTrain = np.array([1 if x > 0.5 else 0 for x in predictedYTrain]).reshape((trainY.size, 1))
        predictVsTrain = (predictedYTrain == trainY)

        # find accuracy of test prediction
        testNumCorrect = np.sum(predictVsTest)
        testAccuracyList.append( testNumCorrect / len(testY) )
        # find accuracy of train prediction
        trainNumCorrect = np.sum(predictVsTrain)
        trainAccuracyList.append( trainNumCorrect / len(trainY) )
        
        # find precision and recall on test prediction
        addingList = list( map(add, predictedYTest, testY) )
        testTruePositives = addingList.count(2)
        subbingList = list( map(sub, testY, predictedYTest))
        testFalsePositives = subbingList.count(-1)
        subbingList2 = list( map(sub, predictedYTest, testY))
        testFalseNegatives = subbingList2.count(-1)
        testTrueNegatives = addingList.count(0)
        testPrecision = testTruePositives / (testTruePositives + testFalsePositives + 10e-10)
        testPrecisionList.append(testPrecision)
        testRecall = testTruePositives / (testTruePositives + testFalseNegatives + 10e-10)
        testRecallList.append(testRecall)
        
        # find precision and recall on train prediction
        addingList = list( map(add, predictedYTrain, trainY) )
        trainTruePositives = addingList.count(2)
        subbingList = list( map(sub, trainY, predictedYTrain))
        trainFalsePositives = subbingList.count(-1)
        subbingList2 = list( map(sub, predictedYTrain, trainY))
        trainFalseNegatives = subbingList2.count(-1)
        trainTrueNegatives = addingList.count(0)
        trainPrecision = trainTruePositives / (trainTruePositives + trainFalsePositives + 10e-10)
        trainPrecisionList.append(trainPrecision)
        trainRecall = trainTruePositives / (trainTruePositives + trainFalseNegatives + 10e-10)
        trainRecallList.append(trainRecall)
    
 	# Validation Accuracy
	# Validation Precision
	# Validation Recall
	# Train Accuracy
	# Train Precision
	# Train Recall
    print("")
    print("" + str(sum(testAccuracyList)/len(testAccuracyList)) )
    print("" + str(sum(testPrecisionList)/len(testPrecisionList)) )
    print("" + str(sum(testRecallList)/len(testRecallList)) )
    print("" + str(sum(trainAccuracyList)/len(trainAccuracyList)) )
    print("" + str(sum(trainPrecisionList)/len(trainPrecisionList)) )
    print("" + str(sum(trainRecallList)/len(trainRecallList)) )
    print("")



# 0.2280598838175628
# 0.481595755771974
# 0.23558628724480069
# 0.2301726375370053
# 0.48846988037097355
# 0.23344679261479645
#
# model = LogisticRegression(penalty='l1')

# 0.22877590371765444
# 0.46821720549384915
# 0.20625645592994726
# 0.22824358761092967
# 0.4636744510295876
# 0.2070069345007207
#
# model = LogisticRegression(penalty='l2')



# 0.23021436750725757
# 0.48488752660232776
# 0.22859493317177987
# 0.23056631168564898
# 0.48888157352297823
# 0.23354161478059116
#
# model = LogisticRegression(penalty='l1', tol=1e-6)

# 0.22835008499039972
# 0.4953487560969211
# 0.22825825463408816
# 0.2311017769562394
# 0.488894822442764
# 0.23473547079025603
#
#model = LogisticRegression(penalty='l1', tol=1e-5)

# 0.2287774332389449
# 0.4832681753684448
# 0.22800509669351404
# 0.23060193660131026
# 0.48997063753129233
# 0.23723966983948763
#
# model = LogisticRegression(penalty='l1', tol=1e-4)

# 0.22692202194557148
# 0.4840317612791424
# 0.22598348965043033
# 0.22978003577428371
# 0.48537367165864265
# 0.2269404391530398
#
# model = LogisticRegression(penalty='l1', tol=1e-3)

# 0.22477426814955453
# 0.4463711942417657
# 0.18818871433394463
# 0.22770737550014358
# 0.44433764458408886
# 0.19467884883188144
#
# model = LogisticRegression(penalty='l1', tol=1e-2)

# 0.21663874440577588
# 0.2610983102874803
# 0.08830017746627179
# 0.2172409335834376
# 0.2513593027186906
# 0.08476138480608915
#
# model = LogisticRegression(penalty='l1', tol=1e-1)

# 0.21020098929437067
# 0.08041467304423286
# 0.02209813724056513
# 0.2078102951857149
# 0.08151365523221607
# 0.022744717822149325
#
# model = LogisticRegression(penalty='l1', tol=1e-0)



# 0.12961295973586187
# 0.25271138416568784
# 0.22133882362681234
# 0.1330364122256591
# 0.2558831456531835
# 0.23285799764321152
#
# model = LogisticRegression(penalty='l1', tol=1e-5, class_weight='balanced')


######################################################
#‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’
#######################################################

# 0.2290697757415884
# 0.48620957683757615
# 0.21777957571663284
# 0.22999460870744176
# 0.4806907275266129
# 0.2202353791033695
# model = LogisticRegression(penalty='l2', tol=1e-5, solver='newton-cg')

# 0.2262105906093511
# 0.4688219092253714
# 0.21014937304272507
# 0.23003042512058913
# 0.4756703865300856
# 0.2199678034491804
#
# model = LogisticRegression(penalty='l2', tol=1e-5, solver='lbfgs')

# 0.22863692121639767
# 0.49376409945930994
# 0.23787605289483155
# 0.23060166850482977
# 0.4951715080246374
# 0.23672370511530022
#
# model = LogisticRegression(penalty='l1', tol=1e-5, solver='liblinear')

# 0.2279178422737252
# 0.47940713079094177
# 0.21620505877612578
# 0.22935096650696202
# 0.4794099158543226
# 0.22003007652554665
#
# model = LogisticRegression(penalty='l2', tol=1e-5, solver='sag')

# 0.2276395713669536
# 0.4805623383540956
# 0.2326611785734008
# 0.23031605638758376
# 0.4960846672661298
# 0.23801198598268275
# 
# model = LogisticRegression(penalty='l1', tol=1e-5, solver='saga')











# kernels: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’

# 0.21793333122599287
# 0.28511087126672374
# 0.09333710620586154
# 0.21909919327215716
# 0.28825166317903184
# 0.0945489855681301
# model = svm.SVC(kernel='linear')


# 0.21406058331863295
# 0.0
# 0.0
# 0.21406112421024842
# 0.0
# 0.0
# model = svm.SVC(kernel='poly')


# 0.21691946254661215
# 0.17411508902242487
# 0.04962429610847598
# 0.2177051873222026
# 0.1845107897360932
# 0.053806332994537985
# model = svm.SVC(kernel='rbf')


# 0.2167841508964524
# 0.13593111693519863
# 0.037617338169550305
# 0.21699109957984178
# 0.13936869933342502
# 0.03819972780852261
# model = svm.SVC(kernel='sigmoid')


i = 1000
#for i in range(10, 100, 10):
print("i = " + str(i))
model = RandomForestClassifier(n_estimators=i)
# max_depth=None, random_state=0
fiveFold(model, np.concatenate((trainX, trainY), axis=1))



# 0.1888991423464284
# 0.5068513472475396
# 0.4728453352339237
# 0.9854955910257128
# 0.9970909890747606
# 0.9918308620905725
# model = RandomForestClassifier()

#fiveFold(model, np.concatenate((trainX, trainY), axis=1))





#fiveFold(model, np.concatenate((trainX, trainY), axis=1))






####################################################################
# FOR ACCURACY, DON'T NEED TO FIT MODEL, CROSS VAL DOES IT FOR YOU
####################################################################

# from sklearn.model_selection import cross_val_score
# import sklearn.metrics # import accuracy_score, precision_score, recall_score

# y = np.reshape(cleanedData[:, 0], (numInstances, 1)).astype(int).flatten()
# # for i in range(10,15):
#     #clf = RandomForestClassifier(n_estimators=200, max_depth=i, random_state=0)
# scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
# print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# ######################
# # PREDICTING PORTION
# ######################

# model.fit(X,y)

# testData = pd.read_csv('ItsPersonal/data/test_data.csv', encoding="ISO-8859-1", engine='python')
# testCleanedData = testData.values

# numTestInstances = len(testData.values)
# numTestFeatures = len(testData.values[0])

# testX = testCleanedData[:, 2:numTestFeatures].tolist()
# testYActual = np.reshape(testCleanedData[:, 0], (numTestInstances, 1)).astype(int).flatten()
# testYPredict = model.predict(testX)


# ##########################################################################################
# # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
# ##########################################################################################

# accuracyScore = sklearn.metrics.accuracy_score(testYActual, testYPredict)
# print("Accuracy: " + str(accuracyScore.mean()))

# precisionScore = sklearn.metrics.precision_score(testYActual, testYPredict, average='weighted')
# print("Precision: " + str(precisionScore.mean()))
# # print("Precision: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# recallScore = sklearn.metrics.recall_score(testYActual, testYPredict, average='weighted')
# print("Recall: " + str(recallScore.mean()))



