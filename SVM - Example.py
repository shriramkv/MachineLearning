#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np

# The cancer dataset is to be imported from the scikit-learn dataset library. 
# Import scikit-learn dataset library

from sklearn import datasets

#This is to be done to import the cancer dataset. 
#What do we do now is to load the the dataset.  

cancer = datasets.load_breast_cancer()

# print the names of the 30 features
print("Features: ", cancer.feature_names)

# print the label type of cancer('malignant' 'benign')
print("Labels: ", cancer.target_names)

# print data(feature)shape
print(cancer.data.shape)

# print the cancer data features (top 5 records)
print(cancer.data[0:5])

# print the cancer labels (0:malignant, 1:benign)
print(cancer.target)

# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2,random_state=109) 
# 80% training and 20% test

#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#There are various types of functions such as linear, polynomial, and radial basis function (RBF). 
#Polynomial and RBF are useful for non-linear hyperplane

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:




