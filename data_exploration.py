# -*- coding: utf-8 -*-
"""
Created on Sun May 19 17:34:11 2019

@author: Kevin
"""

import pandas as pd
import numpy as np
path = ('')
df = pd.read_csv(path + 'train_set0506.csv')
df.columns
df.shape
df.describe()
df.isnull().sum()
df = df.dropna()
(df == '?').sum() #Cat7 is a lost cause. More than half of the data is missing
df.Cat7.describe()
df = df.drop(['ID','Household_ID','Cat2','Cat4','Cat5','Cat7'],axis=1)
df= df[df.Cat12.notnull()] #229 rows deleted which has Cat12 data Null
df = df[df != '?'] #Removing rows where the value is equal to '?' 
df.corr() #to see how correlated the attributes are among each other
df.columns
#Too many categories in each attribute, better to drop
df = df.drop(['Blind_Model','Blind_Submodel'],axis=1) 
#Encoding of the Categorical attributes done using dummy variable. 
df = pd.get_dummies(df, columns=['Blind_Make', 'Cat1', 'Cat3',\
                                'Cat6', 'Cat8', 'Cat9', 'Cat10',\
                                'Cat11', 'Cat12','NVCat'])

#standardizing the attributes before applying PCA
y = df['Claim_Amount']
x = df.drop('Claim_Amount', axis=1)

from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pca = PCA(n_components = 5)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data= principalComponents, columns = ['PC1', 'PC2','PC3', 'PC4', 'PC5'])
pca.explained_variance_ratio_

pca = PCA(0.80)
principalComponents = pca.fit_transform(x)
pca.explained_variance_ratio_
