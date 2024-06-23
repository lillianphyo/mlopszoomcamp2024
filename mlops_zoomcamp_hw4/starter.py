#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip freeze | grep scikit-learn')


# In[2]:


get_ipython().system('python -V')


# In[3]:


import pickle
import pandas as pd
import numpy as np


# In[4]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[5]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[6]:


df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')


# In[7]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[8]:


std_dev = np.std(y_pred)
print(std_dev)


# In[9]:


# Create ride_id column
year = 2023
month = 3
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

# Prepare the results dataframe
df_result = pd.DataFrame({
    'ride_id': df['ride_id'],
    'predicted_duration': y_pred
})

# Save the results to a parquet file
output_file = 'predictions.parquet'
df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

# Check the size of the output file
import os
file_size = os.path.getsize(output_file) / (1024 * 1024)  # size in MB
print(file_size)


# In[ ]:




