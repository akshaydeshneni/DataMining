#!/usr/bin/env python
# coding: utf-8

# In[25]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition, datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

os.chdir("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1")

os.getcwd()

FinalEating = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/FinalEating.csv")
FinalNotEating = pd.read_csv("/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase2/FinalNotEating.csv")

FinalDataFrame = FinalEating
FinalDataFrame = FinalDataFrame.append(FinalNotEating)
FinalDataFrame = FinalDataFrame.drop(FinalDataFrame.columns[0:10], axis = 1)
FinalDataFrame = FinalDataFrame.drop(FinalDataFrame.columns[5:13], axis = 1)

FeatureMatrix = FinalDataFrame

FeatureMatrix.to_csv(r'/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase3/FeatureMatrix.csv')

# print(FeatureMatrix)

sc = StandardScaler()

FeatureMatrix_Std = sc.fit_transform(FeatureMatrix)

pca = decomposition.PCA(n_components = 5)

PCA = pd.DataFrame(pca.fit_transform(FeatureMatrix_Std))
PCA = PCA.rename(columns = {0: 'PC1', 1: 'PC2', 2: 'PC3', 3: 'PC4', 4: 'PC5'})

# print(PCA)
# print(pca.explained_variance_ratio_)
# print(sum(pca.explained_variance_ratio_) * 100)

PCA.to_csv(r'/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase3/NewFeatureMatrix.csv')

pcaExplainedVarianceRatio = np.array(pca.explained_variance_ratio_)
var = np.cumsum(np.round(pcaExplainedVarianceRatio, decimals = 3) * 100)
var2 = (pca.explained_variance_ratio_) * 100

PCATranspose = pd.DataFrame()
PCATranspose = PCA.T

plt.gca()

plt.plot(PCATranspose.index, var)
plt.bar(PCATranspose.index, var2)
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance (%)')
plt.title('Principal Components and Explained Variance')
plt.savefig(r'/Users/akshaydeshneni/Desktop/ASU/CSE572-DataMining/Akshay_Deshneni_DataMiningAssignment1/Phase3/ExplainedVariance.png', dpi = 1000)
plt.show()


# In[ ]:





# In[ ]:




