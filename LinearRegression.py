#!/usr/bin/env python
# coding: utf-8

# In[21]:


# We need numpy and pandas. We have imported the same. 
import pandas as pd
import numpy as np

# The next step is to make sure the pandas print decmial values. We have chosen 5 digit precision. 
np.set_printoptions(precision=5, linewidth=120)

# Let us create a dataframe. We know how to do it! 
GPA_Sal_df = pd.read_csv ('C:\Ana\ML_Playlist\data_reg.csv')

# Can we check if the data is displayed properly? 
GPA_Sal_df.head(10)

# Information on the dataset is always handy. Let's print. 
GPA_Sal_df.info()

# Let us import the statsmodel library 
import statsmodels.api as sms

#An intercept is not included by default and should be added by the user
X = sms.add_constant(GPA_Sal_df ['Marks'])
X.head(5)

Y=GPA_Sal_df ['Salary']

#Well, the split happens here. Traning and Testing dataset. 
# First, let us import the train_test_split () 
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X,Y,train_size=0.8, random_state=50)
# 0.8 = 80 percent to train and 20 percent to test! 

the_salary_lm = sms.OLS (train_y, train_X) . fit()
# fit () method does the complete estimate and moves the result to variable the_salary_pred

print (the_salary_lm.params)
# That is it! Now from the result it is to be understood that - For every one percent increase Salary Raises by 3194 Rupees. 


# In[ ]:





# In[ ]:





# In[ ]:




