#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


# In[2]:


df =pd.read_csv("E:\iit class\Machine learning\Cars93.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.notnull()


# # Handling missing Value

# In[9]:


df.fillna(0)


# In[10]:


df.describe()


# In[44]:


df=df.drop_duplicates(keep='first')
df


# # Univarite Analysis

# In[11]:


df.Type.unique()


# In[12]:


k=df.Type.value_counts()
k


# In[36]:


A=df.groupby(by="Type")['Manufacturer'].count()
A


# In[46]:


def min_max_val (col):
    '''to individually check the min and  max values of each col
    '''
    top=df[col].idxmax()
    top_obs=pd.DataFrame(df.loc[top])
    
    bottom=df[col].idxmin()
    bottom_obs=pd.DataFrame(df.loc[bottom])
    
    min_max_obs=pd.concat([top_obs,bottom_obs],axis=1)
    return min_max_obs


# In[48]:


min_max_val ('Max.Price')


# In[56]:


sns.histplot(df['Max.Price'], kde=True)
plt.show()


# In[13]:


cat=df.Type.unique()
val=df.Type.value_counts()
plt.bar(cat,val,color="green")
plt.xlabel("size of cars")
plt.ylabel("No.of vehicle")


# In[14]:


cat=df.Type.unique()
val=df.groupby(by="Type")['Manufacturer'].count()
plt.bar(cat,val,color="green")
plt.xlabel("size of cars")

plt.ylabel("No.of vehicle")


# In[15]:


no_airbags_car= df.AirBags.unique()
no_airbags_car


# In[16]:


df.groupby(by="AirBags")['Manufacturer'].count()


# In[17]:


plt.figure(figsize=(15,10))
counts = df['Manufacturer'].value_counts()
counts.plot(kind="pie", autopct='%1.1f%%',pctdistance=.90)
#plt.legend(title="Manufacturers")
plt.axis('off')
plt.show()


# In[18]:


df.Price.hist()


# In[88]:


df.columns


# In[19]:


plt.figure(figsize=(12, 10))
plt.hist(df['Price'], bins=20, color='blue', edgecolor='black')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


# In[127]:


plt.figure(figsize=(18,20))
plt.subplot(2,3,1)
sns.countplot(x='Type', data=df) 
plt.subplot(2,3,2)
sns.countplot(x='Origin',data=df)
plt.subplot(2,3,3)
sns.countplot(x='Cylinders', data=df)
plt.subplot(2,3,4)
sns.countplot(x='Passengers',data=df)
plt.subplot(2,3,5)
sns.countplot(x='AirBags',data=df)
plt.subplot(2,3,6)
sns.countplot(x='DriveTrain',data=df)


# In[20]:


plt.figure(figsize=(50,10))
plt.subplot(2,3,2)
sns.countplot(x='Model',data=df)


# In[135]:


plt.figure(figsize=(60,10))
plt.subplot(2,3,2)
sns.countplot(x='Manufacturer',data=df)


# In[21]:


plt.figure(figsize=(30,10))
plt.subplot(2,3,2)
sns.countplot(x='Man.trans.avail',data=df)


# In[22]:


cat_col=[]
num_col=[]
for i in df.columns:
    if(df[i].dtypes=="object"):
        cat_col.append(i)
    else:
        num_col.append(i)


# In[26]:


cat_col


# In[25]:


num_col
#len(num_col)


# In[28]:


plt.figure(figsize=(30,15))

x = 1
for i in cat_col:
    plt.subplot(3,3,x)
    sns.countplot(x=df[i],data=df)
    x = x+1


# In[152]:


plt.figure(figsize=(30,15))

x = 1
for i in num_col:
    plt.subplot(3,6,x)
    df[i].hist()
    x = x+1


# # Bivariate Analysis

# In[29]:


import matplotlib.pyplot as plt

plt.scatter(df['Price'], df['Weight'])
plt.xlabel("Price of the car")
plt.ylabel("Weight of the car")


# In[58]:


sns.regplot(x='Horsepower',y='Price',data=df)


# In[62]:


sns.regplot(x='EngineSize',y='MPG.city',data=df)


# In[ ]:





# In[30]:


for i in df.columns:
    if df[i].dtype == "object":
        plt.figure(figsize=(10,5)) 
        sns.countplot(x='Origin',hue= i ,data=df)
        plt.ylabel('Count')
        plt.show()


# # Multivariate 

# In[33]:


sns.pairplot(df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




