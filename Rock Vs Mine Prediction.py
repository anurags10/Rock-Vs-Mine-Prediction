#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as n
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import seaborn as sns


# In[8]:


ds = pd.read_csv('sona.csv')
ds.head()


# In[9]:


ds.columns


# In[10]:


ds.isnull().sum()


# In[11]:


ds.shape


# In[17]:


ds['R'].value_counts()


# In[18]:


ds.groupby(ds['R']).mean()


# In[21]:


x = ds.iloc[:,:-1].values
y = ds.iloc[:,-1].values


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y,random_state=1)


# In[28]:


classifier = LogisticRegression()
classifier.fit(X_train,y_train)


# In[32]:


# accuracy score on trainingt data
X_train_prediction = classifier.predict(X_train)
t =accuracy_score(X_train_prediction,y_train)
t


# In[34]:


# accuracy on test data
X_test_prediction = classifier.predict(X_test)
t2 =accuracy_score(X_test_prediction,y_test)
t2


# #  Making predictive system
# 

# In[43]:


input_data =(0.0164,0.0627,0.0738,0.0608,0.0233,0.1048,0.1338,0.0644,0.1522,0.0780,0.1791,0.2681,0.1788,0.1039,0.1980,0.3234,0.3748,0.2586,0.3680,0.3508,0.5606,0.5231,0.5469,0.6954,0.6352,0.6757,0.8499,0.8025,0.6563,0.8591,0.6655,0.5369,0.3118,0.3763,0.2801,0.0875,0.3319,0.4237,0.1801,0.3743,0.4627,0.1614,0.2494,0.3202,0.2265,0.1146,0.0476,0.0943,0.0824,0.0171,0.0244,0.0258,0.0143,0.0226,0.0187,0.0185,0.0110,0.0094,0.0078,0.0112)

# making input_data to numpy array
n_array = n.asarray(input_data)

# reshape the np array as we are predicting for one instances

n_array_reshape = n_array.reshape(1,-1)

prediction = classifier.predict(n_array_reshape)
prediction

if (prediction[0]=='R'):
    print("The Object is Rock")
    
else:
    print("The object is Mine")


# In[ ]:





# In[ ]:




