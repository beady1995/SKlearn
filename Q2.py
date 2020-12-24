#!/usr/bin/env python
# coding: utf-8

# In[52]:


from sklearn.datasets import load_wine,load_iris
import matplotlib.pyplot as plt 
import pandas as pd  
import seaborn as sns 
from sklearn.cluster import KMeans
for datatype in ['wine','iris']:
    if datatype=='wine':
        data=load_wine()
    else:
         data=load_iris()
#wine_data=load_wine()df=pd.DataFrame(data['data'])
    df=pd.DataFrame(data['data'])
    df.head()
    distortions = []
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(df)
        distortions.append(kmeanModel.inertia_)
    plt.figure(figsize=(16,8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
    kmeanModel = KMeans(n_clusters=3)
    kmeanModel.fit(df)
    df['k_means']=kmeanModel.predict(df)
    df['target']=data['target']
    fig, axes = plt.subplots(1, 2, figsize=(16,8))
    axes[0].scatter(df[0], df[1], c=df['target'])
    axes[1].scatter(df[0], df[1], c=df['k_means'], cmap=plt.cm.Set1)
    axes[0].set_title('%s Actual'%datatype, fontsize=18)
    axes[1].set_title('%s K_Means'%datatype, fontsize=18)
    plt.show()

