#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import transforms
import re


# In[29]:


with open('ItsPersonal/data/mbti_1.csv') as fullFile:
        fullLines = fullFile.readlines()
#         finalFile = fullFile.seek(0)

        
# X = np.load(fullFile)


# In[46]:


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

trainX = list()
trainY = list()

for line in fullLines:
    label, instance = line.split(",",1)
    if label != 'type':
        trainY.append(meyersToNumbersDict[label])
        #print(label)
        instanceNoBars = instance.replace("|||", " ")
        instanceNoUrls = re.sub(r'http\S+', "", instanceNoBars)
        instanceNoMeyers = re.sub(r'([E,I,e,i][N,S,n,s][F,T,f,t][P,J,p,j])', "abcd", instanceNoUrls)
        trainX.append(instanceNoMeyers)

#print(trainY)
print(trainX[0])


# In[ ]:





# In[ ]:




