#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
from sklearn import decomposition, datasets
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn import preprocessing


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

## Test Train Split Eating

train_Eating = pd.DataFrame()
train_Eating = FinalEatingUser12
train_Eating = train_Eating.append(FinalEatingUser13)
train_Eating = train_Eating.append(FinalEatingUser14)
train_Eating = train_Eating.append(FinalEatingUser16)
train_Eating = train_Eating.append(FinalEatingUser17)
train_Eating = train_Eating.append(FinalEatingUser18)
train_Eating = train_Eating.append(FinalEatingUser23)
train_Eating = train_Eating.append(FinalEatingUser24)
train_Eating = train_Eating.append(FinalEatingUser25)
train_Eating = train_Eating.append(FinalEatingUser26)
train_Eating = train_Eating.append(FinalEatingUser30)
train_Eating = train_Eating.append(FinalEatingUser31)
train_Eating = train_Eating.append(FinalEatingUser32)
train_Eating = train_Eating.append(FinalEatingUser33)
train_Eating = train_Eating.append(FinalEatingUser34)
train_Eating = train_Eating.append(FinalEatingUser36)
train_Eating = train_Eating.append(FinalEatingUser37)
train_Eating = train_Eating.append(FinalEatingUser38)

test_Eating = pd.DataFrame()
test_Eating = FinalEatingUser9
test_Eating = test_Eating.append(FinalEatingUser10)
test_Eating = test_Eating.append(FinalEatingUser11)
test_Eating = test_Eating.append(FinalEatingUser19)
test_Eating = test_Eating.append(FinalEatingUser21)
test_Eating = test_Eating.append(FinalEatingUser22)
test_Eating = test_Eating.append(FinalEatingUser27)
test_Eating = test_Eating.append(FinalEatingUser28)
test_Eating = test_Eating.append(FinalEatingUser29)
test_Eating = test_Eating.append(FinalEatingUser39)
test_Eating = test_Eating.append(FinalEatingUser40)
test_Eating = test_Eating.append(FinalEatingUser41)

train_Eating = train_Eating.drop(train_Eating.columns[0:10], axis = 1)
train_Eating = train_Eating.drop(train_Eating.columns[5:13], axis = 1)

test_Eating = test_Eating.drop(test_Eating.columns[0:10], axis = 1)
test_Eating = test_Eating.drop(test_Eating.columns[5:13], axis = 1)

# print(train_Eating)
# print(test_Eating)

## Test Train Split for NotEating

train_NotEating = pd.DataFrame()
train_NotEating = FinalNotEatingUser12
train_NotEating = train_NotEating.append(FinalNotEatingUser13)
train_NotEating = train_NotEating.append(FinalNotEatingUser14)
train_NotEating = train_NotEating.append(FinalNotEatingUser16)
train_NotEating = train_NotEating.append(FinalNotEatingUser17)
train_NotEating = train_NotEating.append(FinalNotEatingUser18)
train_NotEating = train_NotEating.append(FinalNotEatingUser23)
train_NotEating = train_NotEating.append(FinalNotEatingUser24)
train_NotEating = train_NotEating.append(FinalNotEatingUser25)
train_NotEating = train_NotEating.append(FinalNotEatingUser26)
train_NotEating = train_NotEating.append(FinalNotEatingUser30)
train_NotEating = train_NotEating.append(FinalNotEatingUser31)
train_NotEating = train_NotEating.append(FinalNotEatingUser32)
train_NotEating = train_NotEating.append(FinalNotEatingUser33)
train_NotEating = train_NotEating.append(FinalNotEatingUser34)
train_NotEating = train_NotEating.append(FinalNotEatingUser36)
train_NotEating = train_NotEating.append(FinalNotEatingUser37)
train_NotEating = train_NotEating.append(FinalNotEatingUser38)

test_NotEating = pd.DataFrame()
test_NotEating = FinalNotEatingUser9
test_NotEating = test_NotEating.append(FinalNotEatingUser10)
test_NotEating = test_NotEating.append(FinalNotEatingUser11)
test_NotEating = test_NotEating.append(FinalNotEatingUser19)
test_NotEating = test_NotEating.append(FinalNotEatingUser21)
test_NotEating = test_NotEating.append(FinalNotEatingUser22)
test_NotEating = test_NotEating.append(FinalNotEatingUser27)
test_NotEating = test_NotEating.append(FinalNotEatingUser28)
test_NotEating = test_NotEating.append(FinalNotEatingUser29)
test_NotEating = test_NotEating.append(FinalNotEatingUser39)
test_NotEating = test_NotEating.append(FinalNotEatingUser40)
test_NotEating = test_NotEating.append(FinalNotEatingUser41)

train_NotEating = train_NotEating.drop(train_NotEating.columns[0:10], axis = 1)
train_NotEating = train_NotEating.drop(train_NotEating.columns[5:13], axis = 1)

test_NotEating = test_NotEating.drop(test_NotEating.columns[0:10], axis = 1)
test_NotEating = test_NotEating.drop(test_NotEating.columns[5:13], axis = 1)

train_NotEating = train_NotEating.sample(n = 197935)
test_NotEating = test_NotEating.sample(n = 104280)

# print(train_NotEating)
# print(test_NotEating)

X_train = pd.DataFrame()
y_train = pd.DataFrame()
X_test = pd.DataFrame()
y_test = pd.DataFrame()

# , 'AMP_FFT6', 'AMP_FFT7', 'AMP_FFT8'

X_train = train_Eating[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                        'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
X_train = X_train.append(train_NotEating[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                        'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']])
y_train = train_Eating['ClassLabel']
y_train = y_train.append(train_NotEating['ClassLabel'])

X_test = test_Eating[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                        'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']]
X_test = X_test.append(test_NotEating[['STD_DEV', 'MEAN', 'MIN', 'MAX', 'RMS', 'AMP_FFT1', 'AMP_FFT2', 
                        'AMP_FFT3', 'AMP_FFT4', 'AMP_FFT5']])
y_test = test_Eating['ClassLabel']
y_test = y_test.append(test_NotEating['ClassLabel'])

# print(X_train)
# print(y_train)
# print(X_test)
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

PCA.to_csv(r'/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/Phase2_PCA_Training_NewFeatureMatrix.csv')

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

PCA.to_csv(r'/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment2/Phase2_PCA_Testing_NewFeatureMatrix.csv')

xtrain = PCA_Training_NewFeatureMatrix
xtest =  PCA_Testing_NewFeatureMatrix

y_train = y_train.reset_index(drop = True)
y_test = y_test.reset_index(drop = True)

ytrain = y_train
ytest = y_test

idx1 = np.random.permutation(xtrain.index)
xtrainfinal = xtrain.reindex(idx1, axis = 0)
ytrainfinal = ytrain.reindex(idx1, axis = 0)

idx2 = np.random.permutation(xtest.index)
xtestfinal = xtest.reindex(idx2, axis = 0)
ytestfinal = ytest.reindex(idx2, axis = 0)

xtrainfinal = xtrainfinal.reset_index(drop = True)
ytrainfinal = ytrainfinal.reset_index(drop = True)
xtestfinal = xtestfinal.reset_index(drop = True)
ytestfinal = ytestfinal.reset_index(drop = True)

# print(xtrainfinal)
# print(ytrainfinal)
# print(xtestfinal)
# print(ytestfinal)

### Decision Tree

clf_tree = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth= 2,
                       max_features= 'auto' , max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split= 25,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state= 15, splitter='best')

clf_tree.fit(xtrainfinal, ytrainfinal)
y_pred_tree = clf_tree.predict(xtestfinal)

Accuracy_of_DecisionTree = metrics.accuracy_score(ytestfinal, y_pred_tree)
Precision_of_DecisionTree = metrics.precision_score(ytestfinal, y_pred_tree, pos_label = 'Eating')
Recall_of_DecisionTree = metrics.recall_score(ytestfinal, y_pred_tree, pos_label = 'Eating')
F1Score_of_DecisionTree = metrics.f1_score(ytestfinal, y_pred_tree, pos_label = 'Eating')

print('Decision Tree')
print('Accuracy of Decision Tree: ', Accuracy_of_DecisionTree)
print('Precision of Decision Tree: ', Precision_of_DecisionTree)
print('Recall of Decision Tree: ', Recall_of_DecisionTree) 
print('F1 Score of DecisionTree: ', F1Score_of_DecisionTree)
print(metrics.classification_report(ytestfinal, y_pred_tree))
print('Confusion Matrix: ')
print(metrics.confusion_matrix(ytestfinal, y_pred_tree))
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

clf_neural.fit(xtrainfinal, np.ravel(ytrainfinal, order = 'C'))

y_pred_neural = clf_neural.predict(xtestfinal)

Accuracy_of_Neural = metrics.accuracy_score(ytestfinal, y_pred_neural)
Precision_of_Neural = metrics.precision_score(ytestfinal, y_pred_neural, pos_label = 'Eating')
Recall_of_Neural = metrics.recall_score(ytestfinal, y_pred_neural, pos_label = 'Eating')
F1Score_of_Neural = metrics.f1_score(ytestfinal, y_pred_neural, pos_label = 'Eating')

print('Neural Network - Multi layer Perceptron (MLP)')
print('Accuracy of Neural Network: ', Accuracy_of_Neural)
print('Precision of Neural Network: ', Precision_of_Neural)
print('Recall of Neural Network: ', Recall_of_Neural)
print('F1 Score of Neural Network: ', F1Score_of_Neural)
print(metrics.classification_report(ytestfinal, y_pred_neural))
print('Confusion Matrix: ')
print(metrics.confusion_matrix(ytestfinal, y_pred_neural))
print()

### Support Vector Machine

clf_svm = BaggingClassifier(SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                           decision_function_shape='ovr', degree=3,
                           gamma='auto_deprecated', kernel='rbf', max_iter=-1,
                           probability=False, random_state=None, shrinking=True,
                           tol=0.001, verbose=False), n_jobs=-1)

clf_svm.fit(xtrainfinal, np.ravel(ytrainfinal, order = 'C'))

y_pred_svm = clf_svm.predict(xtestfinal)

Accuracy_of_SVM = metrics.accuracy_score(ytestfinal, y_pred_svm)
Precision_of_SVM = metrics.precision_score(ytestfinal, y_pred_svm, pos_label = 'Eating')
Recall_of_SVM = metrics.recall_score(ytestfinal, y_pred_svm, pos_label = 'Eating')
F1Score_of_SVM = metrics.f1_score(ytestfinal, y_pred_svm, pos_label = 'Eating')

print('Support Vector Machine')
print('Accuracy of SVM: ', Accuracy_of_SVM)
print('Precision of SVM: ', Precision_of_SVM)
print('Recall of SVM: ', Recall_of_SVM)
print('F1 Score of SVM: ', F1Score_of_SVM)
print(metrics.classification_report(ytestfinal, y_pred_svm))
print('Confusion Matrix: ')
print(metrics.confusion_matrix(ytestfinal, y_pred_svm))
print()




# In[ ]:





# In[ ]:




