
# coding: utf-8

# In[43]:

# Naive Bayes Classification 
# 
from sklearn.feature_extraction.text import CountVectorizer
import csv
import numpy as np
import random

mbti = []
text = []
mbti_test = []
text_test = []

meyersToNumbersDict = {
    "ISTJ": 0,
    "INTP": 1,
    "ISFJ": 2,
    "INFJ": 3,
    "ISTP": 4,
    "ISFP": 5,
    "INFP": 6,
    "INTJ": 7,
    "ESTP": 8,
    "ESTJ": 9,
    "ESFJ": 10,
    "ENFJ": 11,
    "ESFP": 12,
    "ENTJ": 13,
    "ENTP": 14,
    "ENFP": 15
}

with open("/Users/hannah/Cis 419/Project/train.csv", encoding="utf8", errors='ignore') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    line = 0
    for row in reader:
        if(line == 0):
            line = line + 1          
        else:
            mbti.append(row[0])
            text.append(row[1])
            line = line + 1  

vectorizer = CountVectorizer(max_features = 2500, stop_words = "english", ngram_range = (1,2))
X = vectorizer.fit_transform(text)
arr = X.toarray()


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(arr, mbti)

predictions = clf.predict(arr)
print(clf.score(X, mbti))

with open("/Users/hannah/Cis 419/Project/mbti_5_1_emotion.csv", encoding="utf8", errors='ignore') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    line = 0
    for row in reader:
        if(line == 0):
            line = line + 1          
        else:
            mbti_test.append(row[0])
            text_test.append(row[1])
            
X_test = vectorizer.transform(text_test)
arr_test = X_test.toarray()
predictions = clf.predict(arr_test)
print(clf.score(X_test, mbti_test))


# In[ ]:



