#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
import numpy as np
import pandas as pd
from scipy.fftpack import fft, fftfreq
import matplotlib as plt


os.chdir("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1")

os.getcwd()

EatingFork = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase1/FinalEatingFork.csv")
NotEatingFork = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase1/FinalNotEatingFork.csv")
EatingSpoon = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase1/FinalEatingSpoon.csv")
NotEatingSpoon = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase1/FinalNotEatingSpoon.csv")

Eating = EatingFork.append(EatingSpoon)
NotEating = NotEatingFork.append(NotEatingSpoon)

FinalEating = EatingFork.append(EatingSpoon)
FinalEating = FinalEating.drop(FinalEating.columns[0], axis =1)
FinalNotEating = NotEatingFork.append(NotEatingSpoon)
FinalNotEating = FinalNotEating.drop(FinalNotEating.columns[0], axis =1)

Eating = Eating.drop(Eating.columns[0:3], axis = 1)
NotEating = NotEating.drop(NotEating.columns[0:3], axis = 1)

# print(Eating.head())
# print(NotEating.head())

Eating['STD_DEV'] = Eating.std(axis = 1)
NotEating['STD_DEV'] = NotEating.std(axis =1)

FinalEating['STD_DEV'] = Eating['STD_DEV']
FinalNotEating['STD_DEV'] = NotEating['STD_DEV']

Eating = Eating.drop(Eating.columns[8], axis = 1)
NotEating = NotEating.drop(NotEating.columns[8], axis = 1)

Eating['MEAN'] = Eating.mean(axis = 1)
NotEating['MEAN'] = NotEating.mean(axis = 1)

FinalEating['MEAN'] = Eating['MEAN']
FinalNotEating['MEAN'] = NotEating['MEAN']

Eating = Eating.drop(Eating.columns[8], axis = 1)
NotEating = NotEating.drop(NotEating.columns[8], axis = 1)

Eating['MIN'] = Eating.min(axis = 1)
NotEating['MIN'] = NotEating.min(axis = 1)

FinalEating['MIN'] = Eating['MIN']
FinalNotEating['MIN'] = NotEating['MIN']

Eating = Eating.drop(Eating.columns[8], axis = 1)
NotEating = NotEating.drop(NotEating.columns[8], axis = 1)

Eating['MAX'] = Eating.max(axis = 1)
NotEating['MAX'] = NotEating.max(axis = 1)

FinalEating['MAX'] = Eating['MAX']
FinalNotEating['MAX'] = NotEating['MAX']

Eating = Eating.drop(Eating.columns[8], axis = 1)
NotEating = NotEating.drop(NotEating.columns[8], axis = 1)

for index, row in Eating.iterrows():
    Eating['RMS'] = np.sqrt(((Eating['EMG1'] ** 2) + (Eating['EMG2'] ** 2) + (Eating['EMG3'] ** 2)
                               + (Eating['EMG4'] ** 2) + (Eating['EMG5'] ** 2) + (Eating['EMG6'] ** 2)
                               + (Eating['EMG7'] ** 2) + (Eating['EMG8'] ** 2))/8)

for index, row in NotEating.iterrows():
    NotEating['RMS'] = np.sqrt(((NotEating['EMG1'] ** 2) + (NotEating['EMG2'] ** 2) + (NotEating['EMG3'] ** 2)
                               + (NotEating['EMG4'] ** 2) + (NotEating['EMG5'] ** 2) + (NotEating['EMG6'] ** 2)
                               + (NotEating['EMG7'] ** 2) + (NotEating['EMG8'] ** 2))/8)

FinalEating['RMS'] = Eating['RMS']
FinalNotEating['RMS'] = NotEating['RMS']

Eating = Eating.drop(Eating.columns[8], axis = 1)
NotEating = NotEating.drop(NotEating.columns[8], axis = 1)

EatingArray = Eating
fftArrayEating = fft(EatingArray, axis = 1)
pd1 = pd.DataFrame(fftArrayEating)
AmplitudeFFTArrayEating = np.abs(fftArrayEating) ** 2
pd3 = pd.DataFrame(AmplitudeFFTArrayEating)


NotEatingArray = NotEating
fftArrayNotEating = fft(NotEatingArray, axis =1)
pd2 = pd.DataFrame(fftArrayNotEating)
AmplitudeFFTArrayNotEating = np.abs(fftArrayNotEating) ** 2
pd4 = pd.DataFrame(AmplitudeFFTArrayNotEating)



FinalEating[8] =  pd1[0]
FinalEating[9] =  pd1[1]
FinalEating[10] = pd1[2]
FinalEating[11] = pd1[3]
FinalEating[12] = pd1[4]
FinalEating[13] = pd1[5]
FinalEating[14] = pd1[6]
FinalEating[15] = pd1[7]
FinalEating[16] = pd3[0]
FinalEating[17] = pd3[1]
FinalEating[18] = pd3[2]
FinalEating[19] = pd3[3]
FinalEating[20] = pd3[4]
FinalEating[21] = pd3[5]
FinalEating[22] = pd3[6]
FinalEating[23] = pd3[7]


FinalEating.rename(columns = {8: 'FFT1', 9: 'FFT2', 10 : 'FFT3', 11 : 'FFT4', 12 : 'FFT5', 13 : 'FFT6', 14: 'FFT7', 15: 'FFT8', 16: 'AMP_FFT1', 17: 'AMP_FFT2', 18: 'AMP_FFT3',
                              19: 'AMP_FFT4', 20: 'AMP_FFT5', 21: 'AMP_FFT6', 22: 'AMP_FFT7', 23: 'AMP_FFT8'}, inplace = True)

FinalNotEating[8] = pd2[0]
FinalNotEating[9] = pd2[1]
FinalNotEating[10] = pd2[2]
FinalNotEating[11] = pd2[3]
FinalNotEating[12] = pd2[4]
FinalNotEating[13] = pd2[5]
FinalNotEating[14] = pd2[6]
FinalNotEating[15] = pd2[7]
FinalNotEating[16] = pd4[0]
FinalNotEating[17] = pd4[1]
FinalNotEating[18] = pd4[2]
FinalNotEating[19] = pd4[3]
FinalNotEating[20] = pd4[4]
FinalNotEating[21] = pd4[5]
FinalNotEating[22] = pd4[6]
FinalNotEating[23] = pd4[7]



FinalNotEating.rename(columns = {8: 'FFT1', 9: 'FFT2', 10 : 'FFT3', 11 : 'FFT4', 12 : 'FFT5', 13 : 'FFT6', 14: 'FFT7', 15: 'FFT8', 16: 'AMP_FFT1', 17: 'AMP_FFT2', 18: 'AMP_FFT3',
                                 19: 'AMP_FFT4', 20: 'AMP_FFT5', 21: 'AMP_FFT6', 22: 'AMP_FFT7', 23: 'AMP_FFT8'}, inplace = True)
                   
FinalEating.to_csv(r'/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/FinalEating.csv')
FinalNotEating.to_csv(r'/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/FinalNotEating.csv')

print(FinalEating)
print(FinalNotEating)
print(FinalEating.dtypes)
print(FinalNotEating.dtypes)


# In[ ]:




