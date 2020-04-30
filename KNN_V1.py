#!/usr/bin/env python
# coding: utf-8

# In[11]:


# Can we import all the modules required.? 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_iris 

# Loading data
irisData = load_iris()

# We Should know the feature names - Print them! 
print("Features: ", irisData.feature_names)

# Can we print the label details? Yes, it is easy too. 
print("Labels: ", irisData.target_names)

# print data(feature)shape
print(irisData.data.shape)

# print the data features, this will help us visualize. 
print(irisData.data[0:30])

print(irisData.target)

# It is important step to create the X and Y! Create feature and target arrays 
X = irisData.data 
y = irisData.target 

# As Usual, we shall split the training and testing % ! 
X_train, X_test, y_train, y_test = train_test_split( 
			X, y, test_size = 0.2, random_state=120) 
# Here, it is 80 percent for training and 20 for testing. 
# Seed value is 120. 

KNN = KNeighborsClassifier(n_neighbors=5) 
# can we use KNN!! Yes, K value is a variable. Now I have 5

KNN.fit(X_train, y_train) 
# We got to fit! use fit and that is it. 

# The next step is to calculate the accuracy of the developed model! Lets do that! 
print("accuracy_KNN",KNN.score(X_test, y_test)) 


# In[ ]:





# In[ ]:




