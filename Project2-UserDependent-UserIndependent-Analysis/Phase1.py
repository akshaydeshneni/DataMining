#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


import os
import numpy as np
import pandas as pd
from sklearn import decomposition, datasets
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier


os.chdir("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2")

os.getcwd()

# Eating Data Sets for all Users with Features Extracted and Adding Class Label

FinalEatingUser9 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user9/FinalEatingUser9.csv")
FinalEatingUser10 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user10/FinalEatingUser10.csv")
FinalEatingUser11 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user11/FinalEatingUser11.csv")
FinalEatingUser12 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user12/FinalEatingUser12.csv")
FinalEatingUser13 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user13/FinalEatingUser13.csv")
FinalEatingUser14 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user14/FinalEatingUser14.csv")
FinalEatingUser16 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user16/FinalEatingUser16.csv")
FinalEatingUser17 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user17/FinalEatingUser17.csv")
FinalEatingUser18 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user18/FinalEatingUser18.csv")
FinalEatingUser19 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user19/FinalEatingUser19.csv")
FinalEatingUser21 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user21/FinalEatingUser21.csv")
FinalEatingUser22 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user22/FinalEatingUser22.csv")
FinalEatingUser23 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user23/FinalEatingUser23.csv")
FinalEatingUser24 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user24/FinalEatingUser24.csv")
FinalEatingUser25 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user25/FinalEatingUser25.csv")
FinalEatingUser26 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user26/FinalEatingUser26.csv")
FinalEatingUser27 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user27/FinalEatingUser27.csv")
FinalEatingUser28 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user28/FinalEatingUser28.csv")
FinalEatingUser29 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user29/FinalEatingUser29.csv")
FinalEatingUser30 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user30/FinalEatingUser30.csv")
FinalEatingUser31 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user31/FinalEatingUser31.csv")
FinalEatingUser32 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user32/FinalEatingUser32.csv")
FinalEatingUser33 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user33/FinalEatingUser33.csv")
FinalEatingUser34 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user34/FinalEatingUser34.csv")
FinalEatingUser36 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user36/FinalEatingUser36.csv")
FinalEatingUser37 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user37/FinalEatingUser37.csv")
FinalEatingUser38 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user38/FinalEatingUser38.csv")
FinalEatingUser39 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user39/FinalEatingUser39.csv")
FinalEatingUser40 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user40/FinalEatingUser40.csv")
FinalEatingUser41 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user41/FinalEatingUser41.csv")

FinalEatingUser9['ClassLabel'] = 'Eating'
FinalEatingUser10['ClassLabel'] = 'Eating'
FinalEatingUser11['ClassLabel'] = 'Eating'
FinalEatingUser12['ClassLabel'] = 'Eating'
FinalEatingUser13['ClassLabel'] = 'Eating'
FinalEatingUser14['ClassLabel'] = 'Eating'
FinalEatingUser16['ClassLabel'] = 'Eating'
FinalEatingUser17['ClassLabel'] = 'Eating'
FinalEatingUser18['ClassLabel'] = 'Eating'
FinalEatingUser19['ClassLabel'] = 'Eating'
FinalEatingUser21['ClassLabel'] = 'Eating'
FinalEatingUser22['ClassLabel'] = 'Eating'
FinalEatingUser23['ClassLabel'] = 'Eating'
FinalEatingUser24['ClassLabel'] = 'Eating'
FinalEatingUser25['ClassLabel'] = 'Eating'
FinalEatingUser26['ClassLabel'] = 'Eating'
FinalEatingUser27['ClassLabel'] = 'Eating'
FinalEatingUser28['ClassLabel'] = 'Eating'
FinalEatingUser29['ClassLabel'] = 'Eating'
FinalEatingUser30['ClassLabel'] = 'Eating'
FinalEatingUser31['ClassLabel'] = 'Eating'
FinalEatingUser32['ClassLabel'] = 'Eating'
FinalEatingUser33['ClassLabel'] = 'Eating'
FinalEatingUser34['ClassLabel'] = 'Eating'
FinalEatingUser36['ClassLabel'] = 'Eating'
FinalEatingUser37['ClassLabel'] = 'Eating'
FinalEatingUser38['ClassLabel'] = 'Eating'
FinalEatingUser39['ClassLabel'] = 'Eating'
FinalEatingUser40['ClassLabel'] = 'Eating'
FinalEatingUser41['ClassLabel'] = 'Eating'

# NotEating Data Sets for all Users with Features Extracted

FinalNotEatingUser9 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user9/FinalNotEatingUser9.csv")
FinalNotEatingUser10 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user10/FinalNotEatingUser10.csv")
FinalNotEatingUser11 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user11/FinalNotEatingUser11.csv")
FinalNotEatingUser12 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user12/FinalNotEatingUser12.csv")
FinalNotEatingUser13 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user13/FinalNotEatingUser13.csv")
FinalNotEatingUser14 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user14/FinalNotEatingUser14.csv")
FinalNotEatingUser16 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user16/FinalNotEatingUser16.csv")
FinalNotEatingUser17 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user17/FinalNotEatingUser17.csv")
FinalNotEatingUser18 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user18/FinalNotEatingUser18.csv")
FinalNotEatingUser19 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user19/FinalNotEatingUser19.csv")
FinalNotEatingUser21 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user21/FinalNotEatingUser21.csv")
FinalNotEatingUser22 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user22/FinalNotEatingUser22.csv")
FinalNotEatingUser23 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user23/FinalNotEatingUser23.csv")
FinalNotEatingUser24 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user24/FinalNotEatingUser24.csv")
FinalNotEatingUser25 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user25/FinalNotEatingUser25.csv")
FinalNotEatingUser26 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user26/FinalNotEatingUser26.csv")
FinalNotEatingUser27 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user27/FinalNotEatingUser27.csv")
FinalNotEatingUser28 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user28/FinalNotEatingUser28.csv")
FinalNotEatingUser29 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user29/FinalNotEatingUser29.csv")
FinalNotEatingUser30 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user30/FinalNotEatingUser30.csv")
FinalNotEatingUser31 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user31/FinalNotEatingUser31.csv")
FinalNotEatingUser32 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user32/FinalNotEatingUser32.csv")
FinalNotEatingUser33 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user33/FinalNotEatingUser33.csv")
FinalNotEatingUser34 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user34/FinalNotEatingUser34.csv")
FinalNotEatingUser36 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user36/FinalNotEatingUser36.csv")
FinalNotEatingUser37 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user37/FinalNotEatingUser37.csv")
FinalNotEatingUser38 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user38/FinalNotEatingUser38.csv")
FinalNotEatingUser39 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user39/FinalNotEatingUser39.csv")
FinalNotEatingUser40 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user40/FinalNotEatingUser40.csv")
FinalNotEatingUser41 = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/user41/FinalNotEatingUser41.csv")

FinalNotEatingUser9['ClassLabel'] = 'NotEating'
FinalNotEatingUser10['ClassLabel'] = 'NotEating'
FinalNotEatingUser11['ClassLabel'] = 'NotEating'
FinalNotEatingUser12['ClassLabel'] = 'NotEating'
FinalNotEatingUser13['ClassLabel'] = 'NotEating'
FinalNotEatingUser14['ClassLabel'] = 'NotEating'
FinalNotEatingUser16['ClassLabel'] = 'NotEating'
FinalNotEatingUser17['ClassLabel'] = 'NotEating'
FinalNotEatingUser18['ClassLabel'] = 'NotEating'
FinalNotEatingUser19['ClassLabel'] = 'NotEating'
FinalNotEatingUser21['ClassLabel'] = 'NotEating'
FinalNotEatingUser22['ClassLabel'] = 'NotEating'
FinalNotEatingUser23['ClassLabel'] = 'NotEating'
FinalNotEatingUser24['ClassLabel'] = 'NotEating'
FinalNotEatingUser25['ClassLabel'] = 'NotEating'
FinalNotEatingUser26['ClassLabel'] = 'NotEating'
FinalNotEatingUser27['ClassLabel'] = 'NotEating'
FinalNotEatingUser28['ClassLabel'] = 'NotEating'
FinalNotEatingUser29['ClassLabel'] = 'NotEating'
FinalNotEatingUser30['ClassLabel'] = 'NotEating'
FinalNotEatingUser31['ClassLabel'] = 'NotEating'
FinalNotEatingUser32['ClassLabel'] = 'NotEating'
FinalNotEatingUser33['ClassLabel'] = 'NotEating'
FinalNotEatingUser34['ClassLabel'] = 'NotEating'
FinalNotEatingUser36['ClassLabel'] = 'NotEating'
FinalNotEatingUser37['ClassLabel'] = 'NotEating'
FinalNotEatingUser38['ClassLabel'] = 'NotEating'
FinalNotEatingUser39['ClassLabel'] = 'NotEating'
FinalNotEatingUser40['ClassLabel'] = 'NotEating'
FinalNotEatingUser41['ClassLabel'] = 'NotEating'

# , 'AMP_FFT6', 'AMP_FFT7', 'AMP_FFT8'

# Test Train Split for Each Eating User

# User 9
XEatingUser9 = FinalEatingUser9[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YEatingUser9 = FinalEatingUser9['ClassLabel']

X_train_EatingUser9, X_test_EatingUser9, y_train_EatingUser9, y_test_EatingUser9 = train_test_split(XEatingUser9, YEatingUser9, test_size = 0.4, random_state = 10)

# User 10
XEatingUser10 = FinalEatingUser10[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YEatingUser10 = FinalEatingUser10['ClassLabel']

X_train_EatingUser10, X_test_EatingUser10, y_train_EatingUser10, y_test_EatingUser10 = train_test_split(XEatingUser10, YEatingUser10, test_size = 0.4, random_state = 10)

# User 11
XEatingUser11 = FinalEatingUser11[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YEatingUser11 = FinalEatingUser11['ClassLabel']

X_train_EatingUser11, X_test_EatingUser11, y_train_EatingUser11, y_test_EatingUser11 = train_test_split(XEatingUser11, YEatingUser11, test_size = 0.4, random_state = 10)

# User 12
XEatingUser12 = FinalEatingUser12[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YEatingUser12 = FinalEatingUser12['ClassLabel']

X_train_EatingUser12, X_test_EatingUser12, y_train_EatingUser12, y_test_EatingUser12 = train_test_split(XEatingUser12, YEatingUser12, test_size = 0.4, random_state = 10)

# User 13
XEatingUser13 = FinalEatingUser13[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YEatingUser13 = FinalEatingUser13['ClassLabel']

X_train_EatingUser13, X_test_EatingUser13, y_train_EatingUser13, y_test_EatingUser13 = train_test_split(XEatingUser13, YEatingUser13, test_size = 0.4, random_state = 10)

# User 14
XEatingUser14 = FinalEatingUser14[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YEatingUser14 = FinalEatingUser14['ClassLabel']

X_train_EatingUser14, X_test_EatingUser14, y_train_EatingUser14, y_test_EatingUser14 = train_test_split(XEatingUser14, YEatingUser14, test_size = 0.4, random_state = 10)

# User 16
XEatingUser16 = FinalEatingUser16[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YEatingUser16 = FinalEatingUser16['ClassLabel']

X_train_EatingUser16, X_test_EatingUser16, y_train_EatingUser16, y_test_EatingUser16 = train_test_split(XEatingUser16, YEatingUser16, test_size = 0.4, random_state = 10)

# User 17
XEatingUser17 = FinalEatingUser17[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YEatingUser17 = FinalEatingUser17['ClassLabel']

X_train_EatingUser17, X_test_EatingUser17, y_train_EatingUser17, y_test_EatingUser17 = train_test_split(XEatingUser17, YEatingUser17, test_size = 0.4, random_state = 10)

# User 18
XEatingUser18 = FinalEatingUser18[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YEatingUser18 = FinalEatingUser18['ClassLabel']

X_train_EatingUser18, X_test_EatingUser18, y_train_EatingUser18, y_test_EatingUser18 = train_test_split(XEatingUser18, YEatingUser18, test_size = 0.4, random_state = 10)

# User 19
XEatingUser19 = FinalEatingUser19[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YEatingUser19 = FinalEatingUser19['ClassLabel']

X_train_EatingUser19, X_test_EatingUser19, y_train_EatingUser19, y_test_EatingUser19 = train_test_split(XEatingUser19, YEatingUser19, test_size = 0.4, random_state = 10)

# User 21
XEatingUser21 = FinalEatingUser21[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YEatingUser21 = FinalEatingUser21['ClassLabel']

X_train_EatingUser21, X_test_EatingUser21, y_train_EatingUser21, y_test_EatingUser21 = train_test_split(XEatingUser21, YEatingUser21, test_size = 0.4, random_state = 10)

# User 22
XEatingUser22 = FinalEatingUser22[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YEatingUser22 = FinalEatingUser22['ClassLabel']

X_train_EatingUser22, X_test_EatingUser22, y_train_EatingUser22, y_test_EatingUser22 = train_test_split(XEatingUser22, YEatingUser22, test_size = 0.4, random_state = 10)

# User 23
XEatingUser23 = FinalEatingUser23[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YEatingUser23 = FinalEatingUser23['ClassLabel']

X_train_EatingUser23, X_test_EatingUser23, y_train_EatingUser23, y_test_EatingUser23 = train_test_split(XEatingUser23, YEatingUser23, test_size = 0.4, random_state = 10)

# User 24
XEatingUser24 = FinalEatingUser24[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YEatingUser24 = FinalEatingUser24['ClassLabel']

X_train_EatingUser24, X_test_EatingUser24, y_train_EatingUser24, y_test_EatingUser24 = train_test_split(XEatingUser24, YEatingUser24, test_size = 0.4, random_state = 10)

# User 25
XEatingUser25 = FinalEatingUser25[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YEatingUser25 = FinalEatingUser25['ClassLabel']

X_train_EatingUser25, X_test_EatingUser25, y_train_EatingUser25, y_test_EatingUser25 = train_test_split(XEatingUser25, YEatingUser25, test_size = 0.4, random_state = 10)

# User 26
XEatingUser26 = FinalEatingUser26[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YEatingUser26 = FinalEatingUser26['ClassLabel']

X_train_EatingUser26, X_test_EatingUser26, y_train_EatingUser26, y_test_EatingUser26 = train_test_split(XEatingUser26, YEatingUser26, test_size = 0.4, random_state = 10)

# User 27
XEatingUser27 = FinalEatingUser27[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YEatingUser27 = FinalEatingUser27['ClassLabel']

X_train_EatingUser27, X_test_EatingUser27, y_train_EatingUser27, y_test_EatingUser27 = train_test_split(XEatingUser27, YEatingUser27, test_size = 0.4, random_state = 10)

# User 28
XEatingUser28 = FinalEatingUser28[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YEatingUser28 = FinalEatingUser28['ClassLabel']

X_train_EatingUser28, X_test_EatingUser28, y_train_EatingUser28, y_test_EatingUser28 = train_test_split(XEatingUser28, YEatingUser28, test_size = 0.4, random_state = 10)

# User 29
XEatingUser29 = FinalEatingUser29[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YEatingUser29 = FinalEatingUser29['ClassLabel']

X_train_EatingUser29, X_test_EatingUser29, y_train_EatingUser29, y_test_EatingUser29 = train_test_split(XEatingUser29, YEatingUser29, test_size = 0.4, random_state = 10)

# User 30
XEatingUser30 = FinalEatingUser30[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YEatingUser30 = FinalEatingUser30['ClassLabel']

X_train_EatingUser30, X_test_EatingUser30, y_train_EatingUser30, y_test_EatingUser30 = train_test_split(XEatingUser30, YEatingUser30, test_size = 0.4, random_state = 10)

# User 31
XEatingUser31 = FinalEatingUser31[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YEatingUser31 = FinalEatingUser31['ClassLabel']

X_train_EatingUser31, X_test_EatingUser31, y_train_EatingUser31, y_test_EatingUser31 = train_test_split(XEatingUser31, YEatingUser31, test_size = 0.4, random_state = 10)

# User 32
XEatingUser32 = FinalEatingUser32[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YEatingUser32 = FinalEatingUser32['ClassLabel']

X_train_EatingUser32, X_test_EatingUser32, y_train_EatingUser32, y_test_EatingUser32 = train_test_split(XEatingUser32, YEatingUser32, test_size = 0.4, random_state = 10)

# User 33
XEatingUser33 = FinalEatingUser33[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YEatingUser33 = FinalEatingUser33['ClassLabel']

X_train_EatingUser33, X_test_EatingUser33, y_train_EatingUser33, y_test_EatingUser33 = train_test_split(XEatingUser33, YEatingUser33, test_size = 0.4, random_state = 10)

# User 34
XEatingUser34 = FinalEatingUser34[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YEatingUser34 = FinalEatingUser34['ClassLabel']

X_train_EatingUser34, X_test_EatingUser34, y_train_EatingUser34, y_test_EatingUser34 = train_test_split(XEatingUser34, YEatingUser34, test_size = 0.4, random_state = 10)

# User 36
XEatingUser36 = FinalEatingUser36[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YEatingUser36 = FinalEatingUser36['ClassLabel']

X_train_EatingUser36, X_test_EatingUser36, y_train_EatingUser36, y_test_EatingUser36 = train_test_split(XEatingUser36, YEatingUser36, test_size = 0.4, random_state = 10)

# User 37
XEatingUser37 = FinalEatingUser37[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YEatingUser37 = FinalEatingUser37['ClassLabel']

X_train_EatingUser37, X_test_EatingUser37, y_train_EatingUser37, y_test_EatingUser37 = train_test_split(XEatingUser37, YEatingUser37, test_size = 0.4, random_state = 10)

# User 38
XEatingUser38 = FinalEatingUser38[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YEatingUser38 = FinalEatingUser38['ClassLabel']

X_train_EatingUser38, X_test_EatingUser38, y_train_EatingUser38, y_test_EatingUser38 = train_test_split(XEatingUser38, YEatingUser38, test_size = 0.4, random_state = 10)

# User 39
XEatingUser39 = FinalEatingUser39[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YEatingUser39 = FinalEatingUser39['ClassLabel']

X_train_EatingUser39, X_test_EatingUser39, y_train_EatingUser39, y_test_EatingUser39 = train_test_split(XEatingUser39, YEatingUser39, test_size = 0.4, random_state = 10)

# User 40
XEatingUser40 = FinalEatingUser40[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YEatingUser40 = FinalEatingUser40['ClassLabel']

X_train_EatingUser40, X_test_EatingUser40, y_train_EatingUser40, y_test_EatingUser40 = train_test_split(XEatingUser40, YEatingUser40, test_size = 0.4, random_state = 10)

# User 41
XEatingUser41 = FinalEatingUser41[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YEatingUser41 = FinalEatingUser41['ClassLabel']

X_train_EatingUser41, X_test_EatingUser41, y_train_EatingUser41, y_test_EatingUser41 = train_test_split(XEatingUser41, YEatingUser41, test_size = 0.4, random_state = 10)

X_train_Eating = pd.DataFrame()
X_train_Eating = X_train_EatingUser9
X_train_Eating = X_train_Eating.append(X_train_EatingUser10)
X_train_Eating = X_train_Eating.append(X_train_EatingUser11)
X_train_Eating = X_train_Eating.append(X_train_EatingUser12)
X_train_Eating = X_train_Eating.append(X_train_EatingUser13)
X_train_Eating = X_train_Eating.append(X_train_EatingUser14)
X_train_Eating = X_train_Eating.append(X_train_EatingUser16)
X_train_Eating = X_train_Eating.append(X_train_EatingUser17)
X_train_Eating = X_train_Eating.append(X_train_EatingUser18)
X_train_Eating = X_train_Eating.append(X_train_EatingUser19)
X_train_Eating = X_train_Eating.append(X_train_EatingUser21)
X_train_Eating = X_train_Eating.append(X_train_EatingUser22)
X_train_Eating = X_train_Eating.append(X_train_EatingUser23)
X_train_Eating = X_train_Eating.append(X_train_EatingUser24)
X_train_Eating = X_train_Eating.append(X_train_EatingUser25)
X_train_Eating = X_train_Eating.append(X_train_EatingUser26)
X_train_Eating = X_train_Eating.append(X_train_EatingUser27)
X_train_Eating = X_train_Eating.append(X_train_EatingUser28)
X_train_Eating = X_train_Eating.append(X_train_EatingUser29)
X_train_Eating = X_train_Eating.append(X_train_EatingUser30)
X_train_Eating = X_train_Eating.append(X_train_EatingUser31)
X_train_Eating = X_train_Eating.append(X_train_EatingUser32)
X_train_Eating = X_train_Eating.append(X_train_EatingUser33)
X_train_Eating = X_train_Eating.append(X_train_EatingUser34)
X_train_Eating = X_train_Eating.append(X_train_EatingUser36)
X_train_Eating = X_train_Eating.append(X_train_EatingUser37)
X_train_Eating = X_train_Eating.append(X_train_EatingUser38)
X_train_Eating = X_train_Eating.append(X_train_EatingUser39)
X_train_Eating = X_train_Eating.append(X_train_EatingUser40)
X_train_Eating = X_train_Eating.append(X_train_EatingUser41)

X_test_Eating = pd.DataFrame()
X_test_Eating = X_test_EatingUser9
X_test_Eating = X_test_Eating.append(X_test_EatingUser10)
X_test_Eating = X_test_Eating.append(X_test_EatingUser11)
X_test_Eating = X_test_Eating.append(X_test_EatingUser12)
X_test_Eating = X_test_Eating.append(X_test_EatingUser13)
X_test_Eating = X_test_Eating.append(X_test_EatingUser14)
X_test_Eating = X_test_Eating.append(X_test_EatingUser16)
X_test_Eating = X_test_Eating.append(X_test_EatingUser17)
X_test_Eating = X_test_Eating.append(X_test_EatingUser18)
X_test_Eating = X_test_Eating.append(X_test_EatingUser19)
X_test_Eating = X_test_Eating.append(X_test_EatingUser21)
X_test_Eating = X_test_Eating.append(X_test_EatingUser22)
X_test_Eating = X_test_Eating.append(X_test_EatingUser23)
X_test_Eating = X_test_Eating.append(X_test_EatingUser24)
X_test_Eating = X_test_Eating.append(X_test_EatingUser25)
X_test_Eating = X_test_Eating.append(X_test_EatingUser26)
X_test_Eating = X_test_Eating.append(X_test_EatingUser27)
X_test_Eating = X_test_Eating.append(X_test_EatingUser28)
X_test_Eating = X_test_Eating.append(X_test_EatingUser29)
X_test_Eating = X_test_Eating.append(X_test_EatingUser30)
X_test_Eating = X_test_Eating.append(X_test_EatingUser31)
X_test_Eating = X_test_Eating.append(X_test_EatingUser32)
X_test_Eating = X_test_Eating.append(X_test_EatingUser33)
X_test_Eating = X_test_Eating.append(X_test_EatingUser34)
X_test_Eating = X_test_Eating.append(X_test_EatingUser36)
X_test_Eating = X_test_Eating.append(X_test_EatingUser37)
X_test_Eating = X_test_Eating.append(X_test_EatingUser38)
X_test_Eating = X_test_Eating.append(X_test_EatingUser39)
X_test_Eating = X_test_Eating.append(X_test_EatingUser40)
X_test_Eating = X_test_Eating.append(X_test_EatingUser41)

y_train_Eating = pd.DataFrame()
y_train_Eating = y_train_EatingUser9
y_train_Eating = y_train_Eating.append(y_train_EatingUser10)
y_train_Eating = y_train_Eating.append(y_train_EatingUser11)
y_train_Eating = y_train_Eating.append(y_train_EatingUser12)
y_train_Eating = y_train_Eating.append(y_train_EatingUser13)
y_train_Eating = y_train_Eating.append(y_train_EatingUser14)
y_train_Eating = y_train_Eating.append(y_train_EatingUser16)
y_train_Eating = y_train_Eating.append(y_train_EatingUser17)
y_train_Eating = y_train_Eating.append(y_train_EatingUser18)
y_train_Eating = y_train_Eating.append(y_train_EatingUser19)
y_train_Eating = y_train_Eating.append(y_train_EatingUser21)
y_train_Eating = y_train_Eating.append(y_train_EatingUser22)
y_train_Eating = y_train_Eating.append(y_train_EatingUser23)
y_train_Eating = y_train_Eating.append(y_train_EatingUser24)
y_train_Eating = y_train_Eating.append(y_train_EatingUser25)
y_train_Eating = y_train_Eating.append(y_train_EatingUser26)
y_train_Eating = y_train_Eating.append(y_train_EatingUser27)
y_train_Eating = y_train_Eating.append(y_train_EatingUser28)
y_train_Eating = y_train_Eating.append(y_train_EatingUser29)
y_train_Eating = y_train_Eating.append(y_train_EatingUser30)
y_train_Eating = y_train_Eating.append(y_train_EatingUser31)
y_train_Eating = y_train_Eating.append(y_train_EatingUser32)
y_train_Eating = y_train_Eating.append(y_train_EatingUser33)
y_train_Eating = y_train_Eating.append(y_train_EatingUser34)
y_train_Eating = y_train_Eating.append(y_train_EatingUser36)
y_train_Eating = y_train_Eating.append(y_train_EatingUser37)
y_train_Eating = y_train_Eating.append(y_train_EatingUser38)
y_train_Eating = y_train_Eating.append(y_train_EatingUser39)
y_train_Eating = y_train_Eating.append(y_train_EatingUser40)
y_train_Eating = y_train_Eating.append(y_train_EatingUser41)

y_test_Eating = pd.DataFrame()
y_test_Eating = y_test_EatingUser9
y_test_Eating = y_test_Eating.append(y_test_EatingUser10)
y_test_Eating = y_test_Eating.append(y_test_EatingUser11)
y_test_Eating = y_test_Eating.append(y_test_EatingUser12)
y_test_Eating = y_test_Eating.append(y_test_EatingUser13)
y_test_Eating = y_test_Eating.append(y_test_EatingUser14)
y_test_Eating = y_test_Eating.append(y_test_EatingUser16)
y_test_Eating = y_test_Eating.append(y_test_EatingUser17)
y_test_Eating = y_test_Eating.append(y_test_EatingUser18)
y_test_Eating = y_test_Eating.append(y_test_EatingUser19)
y_test_Eating = y_test_Eating.append(y_test_EatingUser21)
y_test_Eating = y_test_Eating.append(y_test_EatingUser22)
y_test_Eating = y_test_Eating.append(y_test_EatingUser23)
y_test_Eating = y_test_Eating.append(y_test_EatingUser24)
y_test_Eating = y_test_Eating.append(y_test_EatingUser25)
y_test_Eating = y_test_Eating.append(y_test_EatingUser26)
y_test_Eating = y_test_Eating.append(y_test_EatingUser27)
y_test_Eating = y_test_Eating.append(y_test_EatingUser28)
y_test_Eating = y_test_Eating.append(y_test_EatingUser29)
y_test_Eating = y_test_Eating.append(y_test_EatingUser30)
y_test_Eating = y_test_Eating.append(y_test_EatingUser31)
y_test_Eating = y_test_Eating.append(y_test_EatingUser32)
y_test_Eating = y_test_Eating.append(y_test_EatingUser33)
y_test_Eating = y_test_Eating.append(y_test_EatingUser34)
y_test_Eating = y_test_Eating.append(y_test_EatingUser36)
y_test_Eating = y_test_Eating.append(y_test_EatingUser37)
y_test_Eating = y_test_Eating.append(y_test_EatingUser38)
y_test_Eating = y_test_Eating.append(y_test_EatingUser39)
y_test_Eating = y_test_Eating.append(y_test_EatingUser40)
y_test_Eating = y_test_Eating.append(y_test_EatingUser41)

# Test Train Split for Each NotEating User

# User 9
XNotEatingUser9 = FinalNotEatingUser9[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YNotEatingUser9 = FinalNotEatingUser9['ClassLabel']

X_train_NotEatingUser9, X_test_NotEatingUser9, y_train_NotEatingUser9, y_test_NotEatingUser9 = train_test_split(XNotEatingUser9, YNotEatingUser9, test_size = 0.4, random_state = 10)

# User 10
XNotEatingUser10 = FinalNotEatingUser10[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YNotEatingUser10 = FinalNotEatingUser10['ClassLabel']

X_train_NotEatingUser10, X_test_NotEatingUser10, y_train_NotEatingUser10, y_test_NotEatingUser10 = train_test_split(XNotEatingUser10, YNotEatingUser10, test_size = 0.4, random_state = 10)

# User 11
XNotEatingUser11 = FinalNotEatingUser11[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YNotEatingUser11 = FinalNotEatingUser11['ClassLabel']

X_train_NotEatingUser11, X_test_NotEatingUser11, y_train_NotEatingUser11, y_test_NotEatingUser11 = train_test_split(XNotEatingUser11, YNotEatingUser11, test_size = 0.4, random_state = 10)

# User 12
XNotEatingUser12 = FinalNotEatingUser12[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YNotEatingUser12 = FinalNotEatingUser12['ClassLabel']

X_train_NotEatingUser12, X_test_NotEatingUser12, y_train_NotEatingUser12, y_test_NotEatingUser12 = train_test_split(XNotEatingUser12, YNotEatingUser12, test_size = 0.4, random_state = 10)

# User 13
XNotEatingUser13 = FinalNotEatingUser13[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YNotEatingUser13 = FinalNotEatingUser13['ClassLabel']

X_train_NotEatingUser13, X_test_NotEatingUser13, y_train_NotEatingUser13, y_test_NotEatingUser13 = train_test_split(XNotEatingUser13, YNotEatingUser13, test_size = 0.4, random_state = 10)

# User 14
XNotEatingUser14 = FinalNotEatingUser14[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YNotEatingUser14 = FinalNotEatingUser14['ClassLabel']

X_train_NotEatingUser14, X_test_NotEatingUser14, y_train_NotEatingUser14, y_test_NotEatingUser14 = train_test_split(XNotEatingUser14, YNotEatingUser14, test_size = 0.4, random_state = 10)

# User 16
XNotEatingUser16 = FinalNotEatingUser16[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YNotEatingUser16 = FinalNotEatingUser16['ClassLabel']

X_train_NotEatingUser16, X_test_NotEatingUser16, y_train_NotEatingUser16, y_test_NotEatingUser16 = train_test_split(XNotEatingUser16, YNotEatingUser16, test_size = 0.4, random_state = 10)

# User 17
XNotEatingUser17 = FinalNotEatingUser17[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YNotEatingUser17 = FinalNotEatingUser17['ClassLabel']

X_train_NotEatingUser17, X_test_NotEatingUser17, y_train_NotEatingUser17, y_test_NotEatingUser17 = train_test_split(XNotEatingUser17, YNotEatingUser17, test_size = 0.4, random_state = 10)

# User 18
XNotEatingUser18 = FinalNotEatingUser18[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YNotEatingUser18 = FinalNotEatingUser18['ClassLabel']

X_train_NotEatingUser18, X_test_NotEatingUser18, y_train_NotEatingUser18, y_test_NotEatingUser18 = train_test_split(XNotEatingUser18, YNotEatingUser18, test_size = 0.4, random_state = 10)

# User 19
XNotEatingUser19 = FinalNotEatingUser19[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YNotEatingUser19 = FinalNotEatingUser19['ClassLabel']

X_train_NotEatingUser19, X_test_NotEatingUser19, y_train_NotEatingUser19, y_test_NotEatingUser19 = train_test_split(XNotEatingUser19, YNotEatingUser19, test_size = 0.4, random_state = 10)

# User 21
XNotEatingUser21 = FinalNotEatingUser21[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YNotEatingUser21 = FinalNotEatingUser21['ClassLabel']

X_train_NotEatingUser21, X_test_NotEatingUser21, y_train_NotEatingUser21, y_test_NotEatingUser21 = train_test_split(XNotEatingUser21, YNotEatingUser21, test_size = 0.4, random_state = 10)

# User 22
XNotEatingUser22 = FinalNotEatingUser22[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YNotEatingUser22 = FinalNotEatingUser22['ClassLabel']

X_train_NotEatingUser22, X_test_NotEatingUser22, y_train_NotEatingUser22, y_test_NotEatingUser22 = train_test_split(XNotEatingUser22, YNotEatingUser22, test_size = 0.4, random_state = 10)

# User 23
XNotEatingUser23 = FinalNotEatingUser23[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YNotEatingUser23 = FinalNotEatingUser23['ClassLabel']

X_train_NotEatingUser23, X_test_NotEatingUser23, y_train_NotEatingUser23, y_test_NotEatingUser23 = train_test_split(XNotEatingUser23, YNotEatingUser23, test_size = 0.4, random_state = 10)

# User 24
XNotEatingUser24 = FinalNotEatingUser24[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YNotEatingUser24 = FinalNotEatingUser24['ClassLabel']

X_train_NotEatingUser24, X_test_NotEatingUser24, y_train_NotEatingUser24, y_test_NotEatingUser24 = train_test_split(XNotEatingUser24, YNotEatingUser24, test_size = 0.4, random_state = 10)

# User 25
XNotEatingUser25 = FinalNotEatingUser25[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YNotEatingUser25 = FinalNotEatingUser25['ClassLabel']

X_train_NotEatingUser25, X_test_NotEatingUser25, y_train_NotEatingUser25, y_test_NotEatingUser25 = train_test_split(XNotEatingUser25, YNotEatingUser25, test_size = 0.4, random_state = 10)

# User 26
XNotEatingUser26 = FinalNotEatingUser26[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YNotEatingUser26 = FinalNotEatingUser26['ClassLabel']

X_train_NotEatingUser26, X_test_NotEatingUser26, y_train_NotEatingUser26, y_test_NotEatingUser26 = train_test_split(XNotEatingUser26, YNotEatingUser26, test_size = 0.4, random_state = 10)

# User 27
XNotEatingUser27 = FinalNotEatingUser27[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YNotEatingUser27 = FinalNotEatingUser27['ClassLabel']

X_train_NotEatingUser27, X_test_NotEatingUser27, y_train_NotEatingUser27, y_test_NotEatingUser27 = train_test_split(XNotEatingUser27, YNotEatingUser27, test_size = 0.4, random_state = 10)

# User 28
XNotEatingUser28 = FinalNotEatingUser28[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YNotEatingUser28 = FinalNotEatingUser28['ClassLabel']

X_train_NotEatingUser28, X_test_NotEatingUser28, y_train_NotEatingUser28, y_test_NotEatingUser28 = train_test_split(XNotEatingUser28, YNotEatingUser28, test_size = 0.4, random_state = 10)

# User 29
XNotEatingUser29 = FinalNotEatingUser29[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YNotEatingUser29 = FinalNotEatingUser29['ClassLabel']

X_train_NotEatingUser29, X_test_NotEatingUser29, y_train_NotEatingUser29, y_test_NotEatingUser29 = train_test_split(XNotEatingUser29, YNotEatingUser29, test_size = 0.4, random_state = 10)

# User 30
XNotEatingUser30 = FinalNotEatingUser30[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YNotEatingUser30 = FinalNotEatingUser30['ClassLabel']

X_train_NotEatingUser30, X_test_NotEatingUser30, y_train_NotEatingUser30, y_test_NotEatingUser30 = train_test_split(XNotEatingUser30, YNotEatingUser30, test_size = 0.4, random_state = 10)

# User 31
XNotEatingUser31 = FinalNotEatingUser31[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YNotEatingUser31 = FinalNotEatingUser31['ClassLabel']

X_train_NotEatingUser31, X_test_NotEatingUser31, y_train_NotEatingUser31, y_test_NotEatingUser31 = train_test_split(XNotEatingUser31, YNotEatingUser31, test_size = 0.4, random_state = 10)

# User 32
XNotEatingUser32 = FinalNotEatingUser32[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YNotEatingUser32 = FinalNotEatingUser32['ClassLabel']

X_train_NotEatingUser32, X_test_NotEatingUser32, y_train_NotEatingUser32, y_test_NotEatingUser32 = train_test_split(XNotEatingUser32, YNotEatingUser32, test_size = 0.4, random_state = 10)

# User 33
XNotEatingUser33 = FinalNotEatingUser33[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YNotEatingUser33 = FinalNotEatingUser33['ClassLabel']

X_train_NotEatingUser33, X_test_NotEatingUser33, y_train_NotEatingUser33, y_test_NotEatingUser33 = train_test_split(XNotEatingUser33, YNotEatingUser33, test_size = 0.4, random_state = 10)

# User 34
XNotEatingUser34 = FinalNotEatingUser34[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YNotEatingUser34 = FinalNotEatingUser34['ClassLabel']

X_train_NotEatingUser34, X_test_NotEatingUser34, y_train_NotEatingUser34, y_test_NotEatingUser34 = train_test_split(XNotEatingUser34, YNotEatingUser34, test_size = 0.4, random_state = 10)

# User 36
XNotEatingUser36 = FinalNotEatingUser36[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YNotEatingUser36 = FinalNotEatingUser36['ClassLabel']

X_train_NotEatingUser36, X_test_NotEatingUser36, y_train_NotEatingUser36, y_test_NotEatingUser36 = train_test_split(XNotEatingUser36, YNotEatingUser36, test_size = 0.4, random_state = 10)

# User 37
XNotEatingUser37 = FinalNotEatingUser37[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YNotEatingUser37 = FinalNotEatingUser37['ClassLabel']

X_train_NotEatingUser37, X_test_NotEatingUser37, y_train_NotEatingUser37, y_test_NotEatingUser37 = train_test_split(XNotEatingUser37, YNotEatingUser37, test_size = 0.4, random_state = 10)

# User 38
XNotEatingUser38 = FinalNotEatingUser38[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YNotEatingUser38 = FinalNotEatingUser38['ClassLabel']

X_train_NotEatingUser38, X_test_NotEatingUser38, y_train_NotEatingUser38, y_test_NotEatingUser38 = train_test_split(XNotEatingUser38, YNotEatingUser38, test_size = 0.4, random_state = 10)

# User 39
XNotEatingUser39 = FinalNotEatingUser39[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YNotEatingUser39 = FinalNotEatingUser39['ClassLabel']

X_train_NotEatingUser39, X_test_NotEatingUser39, y_train_NotEatingUser39, y_test_NotEatingUser39 = train_test_split(XNotEatingUser39, YNotEatingUser39, test_size = 0.4, random_state = 10)

# User 40
XNotEatingUser40 = FinalNotEatingUser40[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YNotEatingUser40 = FinalNotEatingUser40['ClassLabel']

X_train_NotEatingUser40, X_test_NotEatingUser40, y_train_NotEatingUser40, y_test_NotEatingUser40 = train_test_split(XNotEatingUser40, YNotEatingUser40, test_size = 0.4, random_state = 10)

# User 41
XNotEatingUser41 = FinalNotEatingUser41[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                                 'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
YNotEatingUser41 = FinalNotEatingUser41['ClassLabel']

X_train_NotEatingUser41, X_test_NotEatingUser41, y_train_NotEatingUser41, y_test_NotEatingUser41 = train_test_split(XNotEatingUser41, YNotEatingUser41, test_size = 0.4, random_state = 10)

X_train_NotEating = pd.DataFrame()
X_train_NotEating = X_train_NotEatingUser9.head(n = 3591)
X_train_NotEating = X_train_NotEating.append(X_train_NotEatingUser10.head(n = 4856 ))
X_train_NotEating = X_train_NotEating.append(X_train_NotEatingUser11.head(n = 2889))
X_train_NotEating = X_train_NotEating.append(X_train_NotEatingUser12.head(n = 4192))
X_train_NotEating = X_train_NotEating.append(X_train_NotEatingUser13.head(n = 6224))
X_train_NotEating = X_train_NotEating.append(X_train_NotEatingUser14.head(n = 6021))
X_train_NotEating = X_train_NotEating.append(X_train_NotEatingUser16.head(n = 9708))
X_train_NotEating = X_train_NotEating.append(X_train_NotEatingUser17.head(n = 13750))
X_train_NotEating = X_train_NotEating.append(X_train_NotEatingUser18.head(n = 7929))
X_train_NotEating = X_train_NotEating.append(X_train_NotEatingUser19.head(n = 5824))
X_train_NotEating = X_train_NotEating.append(X_train_NotEatingUser21.head(n = 9758))
X_train_NotEating = X_train_NotEating.append(X_train_NotEatingUser22.head(n = 5609))
X_train_NotEating = X_train_NotEating.append(X_train_NotEatingUser23.head(n = 5290))
X_train_NotEating = X_train_NotEating.append(X_train_NotEatingUser24.head(n = 10990))
X_train_NotEating = X_train_NotEating.append(X_train_NotEatingUser25.head(n = 2544))
X_train_NotEating = X_train_NotEating.append(X_train_NotEatingUser26.head(n = 3964))
X_train_NotEating = X_train_NotEating.append(X_train_NotEatingUser27.head(n = 5571))
X_train_NotEating = X_train_NotEating.append(X_train_NotEatingUser28.head(n = 5607))
X_train_NotEating = X_train_NotEating.append(X_train_NotEatingUser29.head(n = 4362))
X_train_NotEating = X_train_NotEating.append(X_train_NotEatingUser30.head(n = 7111))
X_train_NotEating = X_train_NotEating.append(X_train_NotEatingUser31.head(n = 6684))
X_train_NotEating = X_train_NotEating.append(X_train_NotEatingUser32.head(n = 4542))
X_train_NotEating = X_train_NotEating.append(X_train_NotEatingUser33.head(n = 10201))
X_train_NotEating = X_train_NotEating.append(X_train_NotEatingUser34.head(n = 4584))
X_train_NotEating = X_train_NotEating.append(X_train_NotEatingUser36.head(n = 5485))
X_train_NotEating = X_train_NotEating.append(X_train_NotEatingUser37.head(n = 3294))
X_train_NotEating = X_train_NotEating.append(X_train_NotEatingUser38.head(n = 6240))
X_train_NotEating = X_train_NotEating.append(X_train_NotEatingUser39.head(n = 2595))
X_train_NotEating = X_train_NotEating.append(X_train_NotEatingUser40.head(n = 3192))
X_train_NotEating = X_train_NotEating.append(X_train_NotEatingUser41.head(n = 8711))

X_test_NotEating = pd.DataFrame()
X_test_NotEating = X_test_NotEatingUser9.head(n = 2394)
X_test_NotEating = X_test_NotEating.append(X_test_NotEatingUser10.head(n = 3238))
X_test_NotEating = X_test_NotEating.append(X_test_NotEatingUser11.head(n = 1927))
X_test_NotEating = X_test_NotEating.append(X_test_NotEatingUser12.head(n = 2796))
X_test_NotEating = X_test_NotEating.append(X_test_NotEatingUser13.head(n = 4150))
X_test_NotEating = X_test_NotEating.append(X_test_NotEatingUser14.head(n = 4014))
X_test_NotEating = X_test_NotEating.append(X_test_NotEatingUser16.head(n = 6472))
X_test_NotEating = X_test_NotEating.append(X_test_NotEatingUser17.head(n = 9167))
X_test_NotEating = X_test_NotEating.append(X_test_NotEatingUser18.head(n = 5286))
X_test_NotEating = X_test_NotEating.append(X_test_NotEatingUser19.head(n = 3883))
X_test_NotEating = X_test_NotEating.append(X_test_NotEatingUser21.head(n = 6506))
X_test_NotEating = X_test_NotEating.append(X_test_NotEatingUser22.head(n = 3740))
X_test_NotEating = X_test_NotEating.append(X_test_NotEatingUser23.head(n = 3527))
X_test_NotEating = X_test_NotEating.append(X_test_NotEatingUser24.head(n = 7328))
X_test_NotEating = X_test_NotEating.append(X_test_NotEatingUser25.head(n = 1697))
X_test_NotEating = X_test_NotEating.append(X_test_NotEatingUser26.head(n = 2643))
X_test_NotEating = X_test_NotEating.append(X_test_NotEatingUser27.head(n = 3714))
X_test_NotEating = X_test_NotEating.append(X_test_NotEatingUser28.head(n = 3738))
X_test_NotEating = X_test_NotEating.append(X_test_NotEatingUser29.head(n = 2909))
X_test_NotEating = X_test_NotEating.append(X_test_NotEatingUser30.head(n = 4742))
X_test_NotEating = X_test_NotEating.append(X_test_NotEatingUser31.head(n = 4457))
X_test_NotEating = X_test_NotEating.append(X_test_NotEatingUser32.head(n = 3029))
X_test_NotEating = X_test_NotEating.append(X_test_NotEatingUser33.head(n = 6802))
X_test_NotEating = X_test_NotEating.append(X_test_NotEatingUser34.head(n = 3056))
X_test_NotEating = X_test_NotEating.append(X_test_NotEatingUser36.head(n = 3658))
X_test_NotEating = X_test_NotEating.append(X_test_NotEatingUser37.head(n = 2197))
X_test_NotEating = X_test_NotEating.append(X_test_NotEatingUser38.head(n = 4161))
X_test_NotEating = X_test_NotEating.append(X_test_NotEatingUser39.head(n = 1730))
X_test_NotEating = X_test_NotEating.append(X_test_NotEatingUser40.head(n = 2128))
X_test_NotEating = X_test_NotEating.append(X_test_NotEatingUser41.head(n = 5808))

y_train_NotEating = pd.DataFrame()
y_train_NotEating = y_train_NotEatingUser9.head(n = 3591)
y_train_NotEating = y_train_NotEating.append(y_train_NotEatingUser10.head(n = 4856))
y_train_NotEating = y_train_NotEating.append(y_train_NotEatingUser11.head(n = 2889))
y_train_NotEating = y_train_NotEating.append(y_train_NotEatingUser12.head(n = 4192))
y_train_NotEating = y_train_NotEating.append(y_train_NotEatingUser13.head(n = 6224))
y_train_NotEating = y_train_NotEating.append(y_train_NotEatingUser14.head(n = 6021))
y_train_NotEating = y_train_NotEating.append(y_train_NotEatingUser16.head(n = 9708))
y_train_NotEating = y_train_NotEating.append(y_train_NotEatingUser17.head(n = 13750))
y_train_NotEating = y_train_NotEating.append(y_train_NotEatingUser18.head(n = 7929))
y_train_NotEating = y_train_NotEating.append(y_train_NotEatingUser19.head(n = 5824))
y_train_NotEating = y_train_NotEating.append(y_train_NotEatingUser21.head(n = 9758))
y_train_NotEating = y_train_NotEating.append(y_train_NotEatingUser22.head(n = 5609))
y_train_NotEating = y_train_NotEating.append(y_train_NotEatingUser23.head(n = 5290))
y_train_NotEating = y_train_NotEating.append(y_train_NotEatingUser24.head(n = 10990))
y_train_NotEating = y_train_NotEating.append(y_train_NotEatingUser25.head(n = 2544))
y_train_NotEating = y_train_NotEating.append(y_train_NotEatingUser26.head(n = 3964))
y_train_NotEating = y_train_NotEating.append(y_train_NotEatingUser27.head(n = 5571))
y_train_NotEating = y_train_NotEating.append(y_train_NotEatingUser28.head(n = 5607))
y_train_NotEating = y_train_NotEating.append(y_train_NotEatingUser29.head(n = 4362))
y_train_NotEating = y_train_NotEating.append(y_train_NotEatingUser30.head(n = 7111))
y_train_NotEating = y_train_NotEating.append(y_train_NotEatingUser31.head(n = 6684))
y_train_NotEating = y_train_NotEating.append(y_train_NotEatingUser32.head(n = 4542))
y_train_NotEating = y_train_NotEating.append(y_train_NotEatingUser33.head(n = 10201))
y_train_NotEating = y_train_NotEating.append(y_train_NotEatingUser34.head(n = 4584))
y_train_NotEating = y_train_NotEating.append(y_train_NotEatingUser36.head(n = 5485))
y_train_NotEating = y_train_NotEating.append(y_train_NotEatingUser37.head(n = 3294))
y_train_NotEating = y_train_NotEating.append(y_train_NotEatingUser38.head(n = 6240))
y_train_NotEating = y_train_NotEating.append(y_train_NotEatingUser39.head(n = 2595))
y_train_NotEating = y_train_NotEating.append(y_train_NotEatingUser40.head(n = 3192))
y_train_NotEating = y_train_NotEating.append(y_train_NotEatingUser41.head(n = 8711))

y_test_NotEating = pd.DataFrame()
y_test_NotEating = y_test_NotEatingUser9.head(n = 2394)
y_test_NotEating = y_test_NotEating.append(y_test_NotEatingUser10.head(n = 3238))
y_test_NotEating = y_test_NotEating.append(y_test_NotEatingUser11.head(n = 1927))
y_test_NotEating = y_test_NotEating.append(y_test_NotEatingUser12.head(n = 2796))
y_test_NotEating = y_test_NotEating.append(y_test_NotEatingUser13.head(n = 4150))
y_test_NotEating = y_test_NotEating.append(y_test_NotEatingUser14.head(n = 4014))
y_test_NotEating = y_test_NotEating.append(y_test_NotEatingUser16.head(n = 6472))
y_test_NotEating = y_test_NotEating.append(y_test_NotEatingUser17.head(n = 9167))
y_test_NotEating = y_test_NotEating.append(y_test_NotEatingUser18.head(n = 5286))
y_test_NotEating = y_test_NotEating.append(y_test_NotEatingUser19.head(n = 3883))
y_test_NotEating = y_test_NotEating.append(y_test_NotEatingUser21.head(n = 6506))
y_test_NotEating = y_test_NotEating.append(y_test_NotEatingUser22.head(n = 3740))
y_test_NotEating = y_test_NotEating.append(y_test_NotEatingUser23.head(n = 3527))
y_test_NotEating = y_test_NotEating.append(y_test_NotEatingUser24.head(n = 7328))
y_test_NotEating = y_test_NotEating.append(y_test_NotEatingUser25.head(n = 1697))
y_test_NotEating = y_test_NotEating.append(y_test_NotEatingUser26.head(n = 2643))
y_test_NotEating = y_test_NotEating.append(y_test_NotEatingUser27.head(n = 3714))
y_test_NotEating = y_test_NotEating.append(y_test_NotEatingUser28.head(n = 3738))
y_test_NotEating = y_test_NotEating.append(y_test_NotEatingUser29.head(n = 2909))
y_test_NotEating = y_test_NotEating.append(y_test_NotEatingUser30.head(n = 4742))
y_test_NotEating = y_test_NotEating.append(y_test_NotEatingUser31.head(n = 4457))
y_test_NotEating = y_test_NotEating.append(y_test_NotEatingUser32.head(n = 3029))
y_test_NotEating = y_test_NotEating.append(y_test_NotEatingUser33.head(n = 6802))
y_test_NotEating = y_test_NotEating.append(y_test_NotEatingUser34.head(n = 3056))
y_test_NotEating = y_test_NotEating.append(y_test_NotEatingUser36.head(n = 3658))
y_test_NotEating = y_test_NotEating.append(y_test_NotEatingUser37.head(n = 2197))
y_test_NotEating = y_test_NotEating.append(y_test_NotEatingUser38.head(n = 4161))
y_test_NotEating = y_test_NotEating.append(y_test_NotEatingUser39.head(n = 1730))
y_test_NotEating = y_test_NotEating.append(y_test_NotEatingUser40.head(n = 2128))
y_test_NotEating = y_test_NotEating.append(y_test_NotEatingUser41.head(n = 5808))


# Combined Test Train Split

X_train = pd.DataFrame()
X_train = X_train_Eating
X_train = X_train.append(X_train_NotEating, ignore_index=False)

y_train = pd.DataFrame()
y_train = y_train_Eating
y_train = y_train.append(y_train_NotEating, ignore_index=False)

X_test = pd.DataFrame()
X_test = X_test_Eating
X_test = X_test.append(X_test_NotEating, ignore_index=False)

y_test = pd.DataFrame()
y_test = y_test_Eating
y_test = y_test.append(y_test_NotEating, ignore_index=False)


# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

# PCA for Training Set

sc = StandardScaler()

X_train_Std = sc.fit_transform(X_train)

pca = decomposition.PCA(n_components = 5)

PCA = pd.DataFrame(pca.fit_transform(X_train_Std))
PCA = PCA.rename(columns = {0: 'PC1', 1: 'PC2', 2: 'PC3', 3: 'PC4', 4: 'PC5'})

PCA_Training_NewFeatureMatrix = PCA

# print(PCA_Training_NewFeatureMatrix)
print(PCA)
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_) * 100)

PCA.to_csv(r'/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/Phase1_PCA_Training_NewFeatureMatrix.csv')

# PCA for Testing Set

sc = StandardScaler()

X_test_Std = sc.fit_transform(X_test)

pca = decomposition.PCA(n_components = 5)

PCA = pd.DataFrame(pca.fit_transform(X_test_Std))
PCA = PCA.rename(columns = {0: 'PC1', 1: 'PC2', 2: 'PC3', 3: 'PC4', 4: 'PC5'})

PCA_Testing_NewFeatureMatrix = PCA

# print(PCA_Testing_NewFeatureMatrix)
print(PCA)
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_) * 100)

PCA.to_csv(r'/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/Phase1_PCA_Testing_NewFeatureMatrix.csv')



xtrain = PCA_Training_NewFeatureMatrix
xtest =  PCA_Testing_NewFeatureMatrix
ytrain = y_train
ytest = y_test


### Decision Tree

clf_tree = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth= 2,
                       max_features= 'auto' , max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split= 25,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state= 15, splitter='best')

clf_tree.fit(xtrain, ytrain)
y_pred_tree = clf_tree.predict(xtest)

Accuracy_of_DecisionTree = metrics.accuracy_score(ytest, y_pred_tree)
Precision_of_DecisionTree = metrics.precision_score(ytest, y_pred_tree, pos_label = 'Eating')
Recall_of_DecisionTree = metrics.recall_score(ytest, y_pred_tree, pos_label = 'Eating')
F1Score_of_DecisionTree = metrics.f1_score(ytest, y_pred_tree, pos_label = 'Eating')

print('Decision Tree')
print('Accuracy of Decision Tree: ', Accuracy_of_DecisionTree)
print('Precision of Decision Tree: ', Precision_of_DecisionTree)
print('Recall of Decision Tree: ', Recall_of_DecisionTree) 
print('F1 Score of DecisionTree: ', F1Score_of_DecisionTree)
print(metrics.classification_report(ytest, y_pred_tree))
print('Confusion Matrix: ')
print(metrics.confusion_matrix(ytest, y_pred_tree))
print()


### Neural Network

clf_neural = BaggingClassifier(MLPClassifier(activation ='relu', alpha=0.001,
                                     batch_size= 'auto', beta_1=0.9, tol = 1e-4,
                                     beta_2=0.999, early_stopping=False,
                                     epsilon=1e-08, hidden_layer_sizes=(50,),
                                     learning_rate='adaptive',
                                     learning_rate_init=0.001, max_iter=200,
                                     momentum=0.9, n_iter_no_change=10,
                                     nesterovs_momentum=True, power_t=0.5, random_state = None, solver = 'adam'), n_jobs=-1)

clf_neural.fit(xtrain, np.ravel(ytrain, order = 'C'))

y_pred_neural = clf_neural.predict(xtest)

Accuracy_of_Neural = metrics.accuracy_score(ytest, y_pred_neural)
Precision_of_Neural = metrics.precision_score(ytest, y_pred_neural, pos_label = 'Eating')
Recall_of_Neural = metrics.recall_score(ytest, y_pred_neural, pos_label = 'Eating')
F1Score_of_Neural = metrics.f1_score(ytest, y_pred_neural, pos_label = 'Eating')

print('Neural Network - Multi layer Perceptron (MLP)')
print('Accuracy of Neural Network: ', Accuracy_of_Neural)
print('Precision of Neural Network: ', Precision_of_Neural)
print('Recall of Neural Network: ', Recall_of_Neural)
print('F1 Score of Neural Network: ', F1Score_of_Neural)
print(metrics.classification_report(ytest, y_pred_neural))
print('Confusion Matrix: ')
print(metrics.confusion_matrix(ytest, y_pred_neural))
print()

### Support Vector Machine

clf_svm = BaggingClassifier(SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                           decision_function_shape='ovr', degree=3,
                           gamma='auto_deprecated', kernel='rbf', max_iter=-1,
                           probability=False, random_state=None, shrinking=True,
                           tol=0.001, verbose=False), n_jobs=-1)

clf_svm.fit(xtrain, np.ravel(ytrain, order = 'C'))

y_pred_svm = clf_svm.predict(xtest)

Accuracy_of_SVM = metrics.accuracy_score(ytest, y_pred_svm)
Precision_of_SVM = metrics.precision_score(ytest, y_pred_svm, pos_label = 'Eating')
Recall_of_SVM = metrics.recall_score(ytest, y_pred_svm, pos_label = 'Eating')
F1Score_of_SVM = metrics.f1_score(ytest, y_pred_svm, pos_label = 'Eating')

print('Support Vector Machine')
print('Accuracy of SVM: ', Accuracy_of_SVM)
print('Precision of SVM: ', Precision_of_SVM)
print('Recall of SVM: ', Recall_of_SVM)
print('F1 Score of SVM: ', F1Score_of_SVM)
print(metrics.classification_report(ytest, y_pred_svm))
print('Confusion Matrix: ')
print(metrics.confusion_matrix(ytest, y_pred_svm))
print()







# In[ ]:





# In[ ]:





# In[ ]:




