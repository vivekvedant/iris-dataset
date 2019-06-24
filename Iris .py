#!/usr/bin/env python
# coding: utf-8

# ## Import Packages

# In[ ]:


import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns


# ## Load Data

# In[2]:


data = pd.read_csv('Iris.csv') 


# ## Preview Data

# In[3]:


data.describe() #statistical details of data


# In[4]:


data.head() #First 5 columns of data


# In[5]:


data['Species'].unique() #unique value of speicies in data


# # visualization of data

# In[6]:


sns.set() #set seaborn


# In[7]:


#drop id from data and save in tmp variable

tmp = data.drop('Id',axis = 1)

#plot each data in tmp by species 

g = sns.pairplot(tmp,hue = 'Species',markers = '+')
plt.show()


# # Create model

# In[9]:


#predictor 

X = data.drop(['Id','Species'],axis= 1)

#target 
y = data['Species']

#print shape of predictor and target

print(X.shape)
print(y.shape)


# In[12]:


#split data and print its shape

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.4,random_state = 5)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[13]:


#Try running from k = 1 through 25 and record testing accuracy

k_range = list(range(1,26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test,y_pred))
    


# In[14]:


#use logisticregression to fit and predict data

logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)

#use accuracy_score metrics to check accuracy of model

print(metrics.accuracy_score(y_test,y_pred))


# In[ ]:




