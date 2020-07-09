#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[11]:


df=pd.read_csv('iris1.csv')


# In[15]:


print(df)


# In[14]:


df.shape


# ## Univariate Analysis

# In[18]:


df_setosa=df.loc[df['species']=='setosa']


# In[21]:


df_virginica=df.loc[df['species']=='virginica']
df_versicolor=df.loc[df['species']=='versicolor']


# In[20]:


df_setosa.shape


# In[34]:


plt.plot(df_setosa['sepal_length'],np.zeros_like(df_setosa['sepal_length']),'o')
plt.plot(df_virginica['sepal_length'],np.zeros_like(df_virginica['sepal_length']),'o')
plt.plot(df_versicolor['sepal_length'],np.zeros_like(df_versicolor['sepal_length']),'o')
plt.xlabel('Sepal Length')
plt.show()


# ## Bivariate Analysis
# 

# In[40]:


sns.FacetGrid(df,hue='species',size=5).map(plt.scatter,'sepal_length','sepal_width').add_legend()


# # Multivariate Analysis

# In[39]:


sns.pairplot(df,hue='species',size=4)


# In[ ]:




