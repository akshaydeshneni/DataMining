#!/usr/bin/env python
# coding: utf-8

# In[31]:


import os
import numpy as np
import pandas as pd

os.chdir("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2")

os.getcwd()

pathFork = "/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/groundTruth/user41/fork/1503959170110.txt"
pathSpoon = "/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/groundTruth/user41/spoon/1503958514022.txt"

grndTruthForkString = np.loadtxt(pathFork, dtype = str, delimiter =',')
grndTruthSpoonString = np.loadtxt(pathSpoon, dtype = str, delimiter = ',')

grndTruthFork = grndTruthForkString.astype(np.int)
grndTruthSpoon = grndTruthSpoonString.astype(np.int)

groundTruthFork = pd.DataFrame({'Start': grndTruthFork[:, 0], 'End': grndTruthFork[:, 1], 'Ignored': grndTruthFork[:, 2]})
groundTruthSpoon = pd.DataFrame({'Start': grndTruthSpoon[:, 0], 'End': grndTruthSpoon[:, 1], 'Ignored': grndTruthSpoon[:, 2]})

# print(groundTruthFork)
# print()
# print(groundTruthSpoon)

groundTruthFork['Start'] = groundTruthFork['Start']* 100
groundTruthFork['Start'] = groundTruthFork['Start'].div(30)

groundTruthFork['End'] = groundTruthFork['End']* 100
groundTruthFork['End'] = groundTruthFork['End'].div(30)

groundTruthSpoon['Start'] = groundTruthSpoon['Start']* 100
groundTruthSpoon['Start'] = groundTruthSpoon['Start'].div(30)

groundTruthSpoon['End'] = groundTruthSpoon['End']* 100
groundTruthSpoon['End'] = groundTruthSpoon['End'].div(30)

# print(groundTruthFork)
# print()
# print(groundTruthSpoon)

groundTruthFork.to_csv(r'/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user41/groundTruthForkCleanedUser41.csv')
groundTruthSpoon.to_csv(r'/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user41/groundTruthSpoonCleanedUser41.csv')


# In[ ]:




