#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import numpy as np
import pandas as pd

os.chdir("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1")

os.getcwd()

EMGDataFork = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/MyoData/user16/fork/1503600471339_EMG.txt", names = ["TimeStamp", "EMG1", "EMG2", "EMG3", "EMG4", "EMG5", "EMG6", "EMG7", "EMG8"])
EMGDataFork.insert(loc = 0, column = 'SampleNumber', value = np.arange(len(EMGDataFork)))

EMGDataSpoon = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/MyoData/user16/spoon/1503599958929_EMG.txt", names = ["TimeStamp", "EMG1", "EMG2", "EMG3", "EMG4", "EMG5", "EMG6", "EMG7", "EMG8"])
EMGDataSpoon.insert(loc = 0, column = 'SampleNumber', value = np.arange(len(EMGDataSpoon)))

groundTruthFork = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/groundTruthForkCleaned.csv")
groundTruthSpoon = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/groundTruthSpoonCleaned.csv")

EMGDataFork.insert(loc = 10, column = 'GroundTruthForkStart', value = groundTruthFork['Start'])
EMGDataFork.insert(loc = 11, column = 'GroundTruthForkEnd', value = groundTruthFork['End'])
EMGDataFork = EMGDataFork.fillna(0)
EMGDataFork.GroundTruthForkStart = EMGDataFork.GroundTruthForkStart.astype(int)
EMGDataFork.GroundTruthForkEnd = EMGDataFork.GroundTruthForkEnd.astype(int)

EMGDataSpoon.insert(loc = 10, column = 'GroundTruthSpoonStart', value = groundTruthSpoon['Start'])
EMGDataSpoon.insert(loc = 11, column = 'GroundTruthSpoonEnd', value = groundTruthSpoon['End'])
EMGDataSpoon = EMGDataSpoon.fillna(0)
EMGDataSpoon.GroundTruthSpoonStart = EMGDataSpoon.GroundTruthSpoonStart.astype(int)
EMGDataSpoon.GroundTruthSpoonEnd = EMGDataSpoon.GroundTruthSpoonEnd.astype(int)

# print(EMGDataFork)
# print(EMGDataSpoon)

column_names = ["SampleNumber", "TimeStamp", "EMG1", "EMG2", "EMG3", "EMG4", "EMG5", "EMG6", "EMG7", "EMG8", "SampleMatchNumber"]

EMGDataEatingFork = pd.DataFrame(columns = column_names)
EMGDataNotEatingFork = pd.DataFrame(columns = column_names)
EMGDataEatingSpoon = pd.DataFrame(columns = column_names)
EMGDataNotEatingSpoon = pd.DataFrame(columns = column_names)

ZippedFork = zip(EMGDataFork['SampleNumber'], EMGDataFork['GroundTruthForkStart'], EMGDataFork['GroundTruthForkEnd'])
ZippedSpoon = zip(EMGDataSpoon['SampleNumber'], EMGDataSpoon['GroundTruthSpoonStart'], EMGDataSpoon['GroundTruthSpoonEnd'])

EatingFork = pd.DataFrame([(i,y) for i, s, e in ZippedFork for y in range(s+1, e-1)], columns = ['SampleNumber', 'SampleMatchNumber'])
EatingSpoon = pd.DataFrame([(i,y) for i, s, e in ZippedSpoon for y in range(s+1, e-1)], columns = ['SampleNumber', 'SampleMatchNumber'])

# print(EatingFork)
# print(EatingSpoon)

# EatingFork.to_csv(r'/Users/akshaydeshneni/Desktop/EatingFork.csv')
# EatingSpoon.to_csv(r'/Users/akshaydeshneni/Desktop/EatingSpoon.csv')

EMGDataFork = EMGDataFork.drop(['GroundTruthForkStart', 'GroundTruthForkEnd'], axis = 1)
EMGDataSpoon = EMGDataSpoon.drop(['GroundTruthSpoonStart', 'GroundTruthSpoonEnd'], axis = 1)

EMGDataFork.insert(loc = 10, column = 'SampleMatchNumber', value = EatingFork['SampleMatchNumber'])
EMGDataSpoon.insert(loc = 10, column = 'SampleMatchNumber', value = EatingSpoon['SampleMatchNumber'])

# print(EMGDataFork.head())
# print(EMGDataSpoon.head())

EMGDataEatingFork = EMGDataFork.loc[EMGDataFork['SampleNumber'].isin(EMGDataFork['SampleMatchNumber'])]
EMGDataNotEatingFork = EMGDataFork.loc[EMGDataFork['SampleNumber'].isin(EMGDataFork['SampleMatchNumber']) == False]
EMGDataEatingSpoon = EMGDataSpoon.loc[EMGDataSpoon['SampleNumber'].isin(EMGDataSpoon['SampleMatchNumber'])]
EMGDataNotEatingSpoon = EMGDataSpoon.loc[EMGDataSpoon['SampleNumber'].isin(EMGDataSpoon['SampleMatchNumber']) == False]

EMGDataEatingFork = EMGDataEatingFork.drop(['SampleMatchNumber'], axis =1)
EMGDataNotEatingFork = EMGDataNotEatingFork.drop(['SampleMatchNumber'], axis =1)
EMGDataEatingSpoon = EMGDataEatingSpoon.drop(['SampleMatchNumber'], axis =1)
EMGDataNotEatingSpoon = EMGDataNotEatingSpoon.drop(['SampleMatchNumber'], axis =1)

EMGDataEatingFork.to_csv(r'/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/FinalEatingFork.csv')
EMGDataNotEatingFork.to_csv(r'/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/FinalNotEatingFork.csv')
EMGDataEatingSpoon.to_csv(r'/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/FinalEatingSpoon.csv')
EMGDataNotEatingSpoon.to_csv(r'/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/FinalNotEatingSpoon.csv')

# print(EMGDataEatingFork.head())
# print("-------------------------")
# print(EMGDataEatingSpoon.head())
# print("-------------------------")
# print(EMGDataNotEatingFork.head())
# print("-------------------------")
# print(EMGDataNotEatingSpoon.head())





# In[ ]:




