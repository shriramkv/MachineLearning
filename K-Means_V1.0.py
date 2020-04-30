#!/usr/bin/env python
# coding: utf-8

# In[33]:


# Can we import all the necessary libraries? 
# Here, you could see, we are importing Label Encoder and MinMax scaler. 
# Both are used at later stage! Stay tuned. 
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#See, we are now referring to a new method of how to access web links 
# We have used train_url and test_url to invoke the test and train dataset. 
# The dataset can as well be downloaded and invoked from the respective path like tradional approach. 
#train_url = "www...../Kaggle/train.csv"
#train = pd.read_csv(train_url)
#test_url = "www.../Kaggle/test.csv"
#test = pd.read_csv(test_url)

# Choose between these two approaches. 
train = pd.read_csv('C:\Ana\ML_Playlist\\train.csv')
test = pd.read_csv('C:\Ana\ML_Playlist\\test.csv')

# It is time to explore. Can we have a look @ the data
print("***** The_Training_DataSet *****")
print(train.head(15))
print(train.describe())
# Can we get some stats? 

print("***** The Testing_DataSet *****")
print(test.head(15))
print(test.describe())
# Can we get some stats?

# Well, here comes an important component for discussion. 
# K-Means cannot handle the missing values in the dataset. Hence, it is important to identify it. 
# After identification, we got to counter it. let us do that now. 

# For the train set, Missing values and summary
train.isna().head()
print('***********************')
print(train.isna().sum())  


# For the test set, Missing values and summary
test.isna().head()
print('***********************')
print(test.isna().sum())  

# We are filling the missed fields with the Mean Value 
train.fillna(train.mean(), inplace=True)
# We are filling the missed fields with the Mean Value 
test.fillna(test.mean(), inplace=True)

# Can we see if the missing values are replaced? 
print('%%%%%%%%%%%%%%%%%%%%%')
print(train.isna().sum())  
print('%%%%%%%%%%%%%%%%%%%%%')
print(test.isna().sum())  

# Can we see how these two features are displayed? I.e. Alphanumeric. 
train['Ticket'].head()
train['Cabin'].head()

# Can we find the survival details with respect to features Pclass, Sex and SibSp. 
print ('&&&&&& PCLASS &&&&&&&&')
print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print ('&&&&&& SEX &&&&&&&&')
print(train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print ('&&&&&& SibSp &&&&&&&&')
print(train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))


# Pclass and Survived features are related and can be seen here.
grid = sns.FacetGrid(train, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();

# Can we drop the Name, Ticket, Cabin and Embarked as they do not have any effect in the final result. 
train = train.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)
test = test.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)

# Sex is the only feature which is to be converted to number. We can use Label Encoder. 
labelEncoder = LabelEncoder()
labelEncoder.fit(train['Sex'])
labelEncoder.fit(test['Sex'])
train['Sex'] = labelEncoder.transform(train['Sex'])
test['Sex'] = labelEncoder.transform(test['Sex'])

# Let us validate the information for numeric conversion. 
print ('Watchout')
print (train.info())
print (test.info())


# We are getting to the core component, Let us drop the survived! 
X = np.array(train.drop(['Survived'], 1).astype(int))
y = np.array(train['Survived'])

# We need two clusters  Correct?? - Alive or Dead. 
kmeans = KMeans(n_clusters=2)  
kmeans.fit(X)

KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=None, tol=0.0001, verbose=0)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))
# Well, we have predicted! 
# We can improve the accuracy!


# In[ ]:





# In[ ]:




