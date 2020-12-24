#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn.datasets import load_boston


# In[5]:


import matplotlib.pyplot as plt 

import pandas as pd  
import seaborn as sns 


# In[6]:


boston_dataset = load_boston()


# In[12]:


boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target
boston.head()


# In[13]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(boston['MEDV'], bins=30)
plt.show()


# In[14]:


correlation_matrix = boston.corr().round(2)
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)


# In[15]:



from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[50]:


X = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
Y = boston['MEDV']


# In[51]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[52]:


lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)


# In[58]:


index=np.argsort(np.abs(lin_model.coef_))


# In[64]:


df = pd.DataFrame(lin_model.coef_[index][::-1].reshape(-1,1), index =X.columns[index][::-1],columns=['impact'])
df.plot.bar()


# In[ ]:




