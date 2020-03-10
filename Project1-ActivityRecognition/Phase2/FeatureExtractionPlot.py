#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import time

os.chdir("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1")

os.getcwd()

FinalEating = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/FinalEating.csv")
FinalNotEating = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/FinalNotEating.csv")

EatingSelected = FinalEating.iloc[1:7001]
EatingSelected = EatingSelected.append(FinalEating.iloc[8987:15987])
EatingSelected['TimeStamp'] = EatingSelected['TimeStamp'].div(1000)
EatingSelected.columns = ['Eating_' + str(col)for col in EatingSelected.columns]


NotEatingSelected = FinalNotEating.iloc[1:7001]
NotEatingSelected = NotEatingSelected.append(FinalNotEating.iloc[30825:37825])
NotEatingSelected['TimeStamp'] = NotEatingSelected['TimeStamp'].div(1000)
NotEatingSelected.columns = ['NotEating_' + str(col)for col in NotEatingSelected.columns]


ax = plt.gca()

# Plot for Eating Standard Deviation vs Time

EatingSelected.plot(kind = 'line', x = 'Eating_TimeStamp' , y = 'Eating_STD_DEV', style = '.-',  ax=ax, color ='red')
plt.xlabel('Eating Time Stamp')
plt.ylabel('Eating Standard Deviation')
plt.title('Eating Standard Deviation vs Time')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/EatingStandardDeviation.png', dpi = 1000)
plt.show()


plt.cla()

# Plot for NotEating Standard Deviation vs Time

NotEatingSelected.plot(kind = 'line', x = 'NotEating_TimeStamp' , y = 'NotEating_STD_DEV', style = '--', ax=ax, color = 'blue')
plt.xlabel('NotEating Time Stamp')
plt.ylabel('NotEating Standard Deviation')
plt.title('NotEating Standard Deviation vs Time')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/NotEatingStandardDeviation.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for Eating Mean vs Time

EatingSelected.plot(kind = 'line', x ='Eating_TimeStamp', y = 'Eating_MEAN', ax=ax, style = '.-', color ='red')
plt.xlabel('Eating Time Stamp')
plt.ylabel('Eating Mean')
plt.title('Eating Mean vs Time', loc = 'center')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/EatingMean.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for NotEating Mean vs Time

NotEatingSelected.plot(kind = 'line', x ='NotEating_TimeStamp', y = 'NotEating_MEAN', ax=ax, style = '--', color = 'blue')
plt.xlabel('NotEating Time Stamp')
plt.ylabel('NotEating Mean')
plt.title('Not Eating Mean vs Time', loc = 'center')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/NotEatingMean.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for Eating Max vs Time

EatingSelected.plot(kind = 'line', x ='Eating_TimeStamp', y = 'Eating_MAX', ax=ax, style = '.-', color ='red')
plt.xlabel('Eating Time Stamp')
plt.ylabel('Eating Max')
plt.title('Eating Max vs Time', loc = 'center')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/EatingMax.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for NotEating Max vs Time

NotEatingSelected.plot(kind = 'line', x ='NotEating_TimeStamp', y = 'NotEating_MAX', ax=ax, style = '--', color = 'blue')
plt.xlabel('NotEating Time Stamp')
plt.ylabel('NotEating Max')
plt.title('NotEating Max vs Time', loc = 'center')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/NotEatingMax.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for Eating Min vs Time

EatingSelected.plot(kind = 'line', x ='Eating_TimeStamp', y = 'Eating_MIN', ax=ax, style = '.-', color ='red')
plt.xlabel('Eating Time Stamp')
plt.ylabel('Eating Min')
plt.title('Eating Min vs Time', loc = 'center')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/EatingMin.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for NotEating Min vs Time

NotEatingSelected.plot(kind = 'line', x ='NotEating_TimeStamp', y = 'NotEating_MIN', ax=ax, style = '--', color = 'blue')
plt.xlabel('NotEating Time Stamp')
plt.ylabel('NotEating Min')
plt.title('NotEating Min vs Time', loc = 'center')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/NotEatingMin.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for Eating RMS vs Time

EatingSelected.plot(kind = 'line', x = 'Eating_TimeStamp', y = 'Eating_RMS', ax=ax, style = '.-', color = 'red')
plt.xlabel('Eating Time Stamp')
plt.ylabel('Eating RMS')
plt.title('Eating RMS vs Time', loc = 'center')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/EatingRMS.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for NotEating RMS vs Time

NotEatingSelected.plot(kind = 'line', x ='NotEating_TimeStamp', y = 'NotEating_RMS', ax=ax, style = '--', color = 'blue')
plt.xlabel('NotEating Time Stamp')
plt.ylabel('NotEating RMS')
plt.title('NotEating RMS vs Time', loc = 'center')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/NotEatingRMS.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for Eating Amplitude of FFT1 vs Time

EatingSelected.plot(kind = 'line', x = 'Eating_TimeStamp', y = 'Eating_AMP_FFT1', ax=ax, style = '.-', color = 'red')
plt.xlabel('Eating Time Stamp')
plt.ylabel('Eating AMP_FFT1')
plt.title('Eating AMP_FFT1 vs Time', loc = 'center')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/Eating_AMP_FFT1.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for Eating Amplitude of FFT2 vs Time

EatingSelected.plot(kind = 'line', x = 'Eating_TimeStamp', y = 'Eating_AMP_FFT2', ax=ax, style = '.-', color = 'red')
plt.xlabel('Eating Time Stamp')
plt.ylabel('Eating AMP_FFT2')
plt.title('Eating AMP_FFT2 vs Time', loc = 'center')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/Eating_AMP_FFT2.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for Eating Amplitude of FFT3 vs Time

EatingSelected.plot(kind = 'line', x = 'Eating_TimeStamp', y = 'Eating_AMP_FFT3', ax=ax, style = '.-', color = 'red')
plt.xlabel('Eating Time Stamp')
plt.ylabel('Eating AMP_FFT3')
plt.title('Eating AMP_FFT3 vs Time', loc = 'center')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/Eating_AMP_FFT3.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for Eating Amplitude of FFT4 vs Time

EatingSelected.plot(kind = 'line', x = 'Eating_TimeStamp', y = 'Eating_AMP_FFT4', ax=ax, style = '.-', color = 'red')
plt.xlabel('Eating Time Stamp')
plt.ylabel('Eating AMP_FFT4')
plt.title('Eating AMP_FFT4 vs Time', loc = 'center')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/Eating_AMP_FFT4.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for Eating Amplitude of FFT5 vs Time

EatingSelected.plot(kind = 'line', x = 'Eating_TimeStamp', y = 'Eating_AMP_FFT5', ax=ax, style = '.-', color = 'red')
plt.xlabel('Eating Time Stamp')
plt.ylabel('Eating AMP_FFT5')
plt.title('Eating AMP_FFT5 vs Time', loc = 'center')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/Eating_AMP_FFT5.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for Eating Amplitude of FFT6 vs Time

EatingSelected.plot(kind = 'line', x = 'Eating_TimeStamp', y = 'Eating_AMP_FFT6', ax=ax, style = '.-', color = 'red')
plt.xlabel('Eating Time Stamp')
plt.ylabel('Eating AMP_FFT6')
plt.title('Eating AMP_FFT6 vs Time', loc = 'center')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/Eating_AMP_FFT6.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for Eating Amplitude of FFT7 vs Time

EatingSelected.plot(kind = 'line', x = 'Eating_TimeStamp', y = 'Eating_AMP_FFT7', ax=ax, style = '.-', color = 'red')
plt.xlabel('Eating Time Stamp')
plt.ylabel('Eating AMP_FFT7')
plt.title('Eating AMP_FFT7 vs Time', loc = 'center')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/Eating_AMP_FFT7.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for Eating Amplitude of FFT8 vs Time

EatingSelected.plot(kind = 'line', x = 'Eating_TimeStamp', y = 'Eating_AMP_FFT8', ax=ax, style = '.-', color = 'red')
plt.xlabel('Eating Time Stamp')
plt.ylabel('Eating AMP_FFT8')
plt.title('Eating AMP_FFT8 vs Time', loc = 'center')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/Eating_AMP_FFT8.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for NotEating Amplitude of FFT1 vs Time

NotEatingSelected.plot(kind = 'line', x = 'NotEating_TimeStamp', y = 'NotEating_AMP_FFT1', ax=ax, style = '.-', color = 'red')
plt.xlabel('NotEating Time Stamp')
plt.ylabel('NotEating AMP_FFT1')
plt.title('NotEating AMP_FFT1 vs Time', loc = 'center')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/NotEating_AMP_FFT1.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for NotEating Amplitude of FFT2 vs Time

NotEatingSelected.plot(kind = 'line', x = 'NotEating_TimeStamp', y = 'NotEating_AMP_FFT2', ax=ax, style = '.-', color = 'red')
plt.xlabel('NotEating Time Stamp')
plt.ylabel('NotEating AMP_FFT2')
plt.title('NotEating AMP_FFT2 vs Time', loc = 'center')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/NotEating_AMP_FFT2.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for NotEating Amplitude of FFT3 vs Time

NotEatingSelected.plot(kind = 'line', x = 'NotEating_TimeStamp', y = 'NotEating_AMP_FFT3', ax=ax, style = '.-', color = 'red')
plt.xlabel('NotEating Time Stamp')
plt.ylabel('NotEating AMP_FFT3')
plt.title('NotEating AMP_FFT3 vs Time', loc = 'center')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/NotEating_AMP_FFT3.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for NotEating Amplitude of FFT4 vs Time

NotEatingSelected.plot(kind = 'line', x = 'NotEating_TimeStamp', y = 'NotEating_AMP_FFT4', ax=ax, style = '.-', color = 'red')
plt.xlabel('NotEating Time Stamp')
plt.ylabel('NotEating AMP_FFT4')
plt.title('NotEating AMP_FFT4 vs Time', loc = 'center')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/NotEating_AMP_FFT4.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for NotEating Amplitude of FFT5 vs Time

NotEatingSelected.plot(kind = 'line', x = 'NotEating_TimeStamp', y = 'NotEating_AMP_FFT5', ax=ax, style = '.-', color = 'red')
plt.xlabel('NotEating Time Stamp')
plt.ylabel('NotEating AMP_FFT5')
plt.title('NotEating AMP_FFT5 vs Time', loc = 'center')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/NotEating_AMP_FFT5.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for NotEating Amplitude of FFT6 vs Time

NotEatingSelected.plot(kind = 'line', x = 'NotEating_TimeStamp', y = 'NotEating_AMP_FFT6', ax=ax, style = '.-', color = 'red')
plt.xlabel('NotEating Time Stamp')
plt.ylabel('NotEating AMP_FFT6')
plt.title('NotEating AMP_FFT6 vs Time', loc = 'center')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/NotEating_AMP_FFT6.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for NotEating Amplitude of FFT7 vs Time

NotEatingSelected.plot(kind = 'line', x = 'NotEating_TimeStamp', y = 'NotEating_AMP_FFT7', ax=ax, style = '.-', color = 'red')
plt.xlabel('NotEating Time Stamp')
plt.ylabel('NotEating AMP_FFT7')
plt.title('NotEating AMP_FFT7 vs Time', loc = 'center')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/NotEating_AMP_FFT7.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for NotEating Amplitude of FFT8 vs Time

NotEatingSelected.plot(kind = 'line', x = 'NotEating_TimeStamp', y = 'NotEating_AMP_FFT8', ax=ax, style = '.-', color = 'red')
plt.xlabel('NotEating Time Stamp')
plt.ylabel('NotEating AMP_FFT8')
plt.title('NotEating AMP_FFT8 vs Time', loc = 'center')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/NotEating_AMP_FFT8.png', dpi = 1000)
plt.show()

plt.cla()

