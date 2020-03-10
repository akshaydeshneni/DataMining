#!/usr/bin/env python
# coding: utf-8

# In[18]:


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

EatingSelected = FinalEating.iloc[1:8986]
EatingSelected = EatingSelected.append(FinalEating.iloc[8987:16180])
EatingSelected['TimeStamp'] = EatingSelected['TimeStamp'].div(1000)
EatingSelected.columns = ['Eating_' + str(col)for col in EatingSelected.columns]


NotEatingSelected = FinalNotEating.iloc[1:30824]
NotEatingSelected = NotEatingSelected.append(FinalNotEating.iloc[30825:68513])
NotEatingSelected['TimeStamp'] = NotEatingSelected['TimeStamp'].div(1000)
NotEatingSelected.columns = ['NotEating_' + str(col)for col in NotEatingSelected.columns]


ax = plt.gca()

# Plot for Eating EMG1 vs Time

EatingSelected.plot(kind = 'line', x = 'Eating_TimeStamp' , y = 'Eating_EMG1', style = '.-',  ax=ax, color ='red')
plt.xlabel('Eating Time Stamp')
plt.ylabel('Eating EMG1')
plt.title('Eating EMG1 vs Time')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/EatingEMG1.png', dpi = 1000)
plt.show()


plt.cla()

# Plot for Eating EMG2 vs Time

EatingSelected.plot(kind = 'line', x = 'Eating_TimeStamp' , y = 'Eating_EMG2', style = '.-',  ax=ax, color ='red')
plt.xlabel('Eating Time Stamp')
plt.ylabel('Eating EMG2')
plt.title('Eating EMG2 vs Time')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/EatingEMG2.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for Eating EMG3 vs Time

EatingSelected.plot(kind = 'line', x = 'Eating_TimeStamp' , y = 'Eating_EMG3', style = '.-',  ax=ax, color ='red')
plt.xlabel('Eating Time Stamp')
plt.ylabel('Eating EMG3')
plt.title('Eating EMG3 vs Time')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/EatingEMG3.png', dpi = 1000)
plt.show()

# Plot for Eating EMG4 vs Time

EatingSelected.plot(kind = 'line', x = 'Eating_TimeStamp' , y = 'Eating_EMG4', style = '.-',  ax=ax, color ='red')
plt.xlabel('Eating Time Stamp')
plt.ylabel('Eating EMG4')
plt.title('Eating EMG4 vs Time')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/EatingEMG4.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for Eating EMG5 vs Time

EatingSelected.plot(kind = 'line', x = 'Eating_TimeStamp' , y = 'Eating_EMG5', style = '.-',  ax=ax, color ='red')
plt.xlabel('Eating Time Stamp')
plt.ylabel('Eating EMG5')
plt.title('Eating EMG5 vs Time')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/EatingEMG5.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for Eating EMG6 vs Time

EatingSelected.plot(kind = 'line', x = 'Eating_TimeStamp' , y = 'Eating_EMG6', style = '.-',  ax=ax, color ='red')
plt.xlabel('Eating Time Stamp')
plt.ylabel('Eating EMG6')
plt.title('Eating EMG6 vs Time')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/EatingEMG6.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for Eating EMG7 vs Time

EatingSelected.plot(kind = 'line', x = 'Eating_TimeStamp' , y = 'Eating_EMG7', style = '.-',  ax=ax, color ='red')
plt.xlabel('Eating Time Stamp')
plt.ylabel('Eating EMG7')
plt.title('Eating EMG7 vs Time')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/EatingEMG7.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for Eating EMG8 vs Time

EatingSelected.plot(kind = 'line', x = 'Eating_TimeStamp' , y = 'Eating_EMG8', style = '.-',  ax=ax, color ='red')
plt.xlabel('Eating Time Stamp')
plt.ylabel('Eating EMG8')
plt.title('Eating EMG8 vs Time')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/EatingEMG8.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for NotEating EMG1 vs Time

NotEatingSelected.plot(kind = 'line', x = 'NotEating_TimeStamp' , y = 'NotEating_EMG1', style = '.-',  ax=ax, color ='blue')
plt.xlabel('NotEating Time Stamp')
plt.ylabel('NotEating EMG1')
plt.title('NotEating EMG1 vs Time')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/NotEatingEMG1.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for NotEating EMG2 vs Time

NotEatingSelected.plot(kind = 'line', x = 'NotEating_TimeStamp' , y = 'NotEating_EMG2', style = '.-',  ax=ax, color ='blue')
plt.xlabel('NotEating Time Stamp')
plt.ylabel('NotEating EMG2')
plt.title('NotEating EMG2 vs Time')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/NotEatingEMG2.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for NotEating EMG3 vs Time

NotEatingSelected.plot(kind = 'line', x = 'NotEating_TimeStamp' , y = 'NotEating_EMG3', style = '.-',  ax=ax, color ='blue')
plt.xlabel('NotEating Time Stamp')
plt.ylabel('NotEating EMG3')
plt.title('NotEating EMG3 vs Time')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/NotEatingEMG3.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for NotEating EMG4 vs Time

NotEatingSelected.plot(kind = 'line', x = 'NotEating_TimeStamp' , y = 'NotEating_EMG4', style = '.-',  ax=ax, color ='blue')
plt.xlabel('NotEating Time Stamp')
plt.ylabel('NotEating EMG4')
plt.title('NotEating EMG4 vs Time')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/NotEatingEMG4.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for NotEating EMG5 vs Time

NotEatingSelected.plot(kind = 'line', x = 'NotEating_TimeStamp' , y = 'NotEating_EMG5', style = '.-',  ax=ax, color ='blue')
plt.xlabel('NotEating Time Stamp')
plt.ylabel('NotEating EMG5')
plt.title('NotEating EMG5 vs Time')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/NotEatingEMG5.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for NotEating EMG6 vs Time

NotEatingSelected.plot(kind = 'line', x = 'NotEating_TimeStamp' , y = 'NotEating_EMG6', style = '.-',  ax=ax, color ='blue')
plt.xlabel('NotEating Time Stamp')
plt.ylabel('NotEating EMG6')
plt.title('NotEating EMG6 vs Time')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/NotEatingEMG6.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for NotEating EMG7 vs Time

NotEatingSelected.plot(kind = 'line', x = 'NotEating_TimeStamp' , y = 'NotEating_EMG7', style = '.-',  ax=ax, color ='blue')
plt.xlabel('NotEating Time Stamp')
plt.ylabel('NotEating EMG7')
plt.title('NotEating EMG7 vs Time')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/NotEatingEMG7.png', dpi = 1000)
plt.show()

plt.cla()

# Plot for NotEating EMG8 vs Time

NotEatingSelected.plot(kind = 'line', x = 'NotEating_TimeStamp' , y = 'NotEating_EMG8', style = '.-',  ax=ax, color ='blue')
plt.xlabel('NotEating Time Stamp')
plt.ylabel('NotEating EMG8')
plt.title('NotEating EMG8 vs Time')
plt.savefig('/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/NotEatingEMG8.png', dpi = 1000)
plt.show()


# In[ ]:




