#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip list --format=freeze | grep scikit-learn


# In[2]:


get_ipython().system('python -V')


# In[9]:


import pickle
import pandas as pd
import numpy as np


# In[5]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[6]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[7]:


df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')


# In[8]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[10]:


np.std(y_pred)


# In[41]:


df['predictions'] = y_pred


# In[21]:


df['year'] = df['tpep_pickup_datetime'].dt.strftime('%Y')
df['month'] = df['tpep_pickup_datetime'].dt.strftime('%m')


# In[38]:


df['ride_id'] = df["year"].astype(str) + '/' + df["month"].astype(str) + '_' + df.index.astype(str)


# In[43]:


df_result = df[['ride_id','predictions']]


# In[46]:


output_file = './df_result.parquet'
df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)


# In[ ]:




