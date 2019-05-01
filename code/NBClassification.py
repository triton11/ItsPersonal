
# coding: utf-8

# In[10]:

# Naive Bayes Classification 
# 
from sklearn.feature_extraction.text import CountVectorizer
import csv
import numpy as np
import random
import nltk
from sklearn.model_selection import cross_val_score

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

with open("/Users/hannah/Cis 419/Project/train_data.csv", encoding="utf8", errors='ignore') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    line = 0
    for row in reader:
        if(line == 0):
            line = line + 1          
        else:
            mbti.append(row[0])
            text.append(row[1])
            line = line + 1  

stopwords = nltk.corpus.stopwords.words('english')
stopwords.append('really')
stopwords.append('like')
stopwords.append('people')
stopwords.append('time')
stopwords.append('abcd')
stopwords.append('abcds')
stopwords.append('things')
stopwords.append('well')
stopwords.append('way')
stopwords.append('also')
stopwords.append('say')
stopwords.append('want')
stopwords.append('say')
stopwords.append('good')
stopwords.append('see')
stopwords.append('get')
stopwords.append('one')
stopwords.append('would')
stopwords.append('go')
stopwords.append('lot')

vectorizer = CountVectorizer(max_features = 2000, stop_words = stopwords, ngram_range = (1,3))
X = vectorizer.fit_transform(text)
arr = X.toarray()
#print(arr.shape)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(arr, mbti)

predictions = clf.predict(arr)
print("Score on training data: %s" % clf.score(X, mbti))

scores = cross_val_score(clf, arr, mbti, cv=5)
print("Cross val scores: %s" %str(scores)[1:-1])

with open("/Users/hannah/Cis 419/Project/test_data.csv", encoding="utf8", errors='ignore') as csv_file:
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
print("Score on testing data: %s" %clf.score(X_test, mbti_test))

names = vectorizer.get_feature_names()

choice = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
vocab = vectorizer.vocabulary_
key_list = list(vocab.keys()) 
val_list = list(vocab.values()) 

mbti_key = list(meyersToNumbersDict.keys())
mbti_val = list(meyersToNumbersDict.values())

for a in choice:
    q = str(a)
    indices = [i for i, x in enumerate(mbti) if x == q]    
    txts = []
    topten = np.arange(10)
    for i in indices:
        txts.append(arr[i, :])
        total = np.sum(txts, axis = 0)
        topten = np.argsort(total)[-20:]
        
    print("top twenty words for %s type:" % (mbti_key[mbti_val.index(a)]))
    words = []
    s = ","
    for num in topten:
        words.append(key_list[val_list.index(num)])
    print(s.join(words))
    print("\n")


# In[ ]:




# In[ ]:



