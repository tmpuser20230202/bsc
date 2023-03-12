#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import graph_tool.all as gt

from datetime import timedelta

import pickle

import os
import tqdm


# In[ ]:


from bsc import calc_paracomp_bern_with_prior_beta
from cython_normterm_discrete import create_fun_with_mem


# In[ ]:


outdir = './output/kdd2023'

if not os.path.exists(outdir):
    os.mkdir(outdir)


# In[ ]:


ground_truths = pd.read_excel('./data/real/TwitterWorldCup2014/Soccer World Cup 2014 - Ground truth_SpreadSheet.xlsx')


# In[ ]:


ground_truths = ground_truths.loc[ground_truths['Event : {Goal, Yellow card, Red card, Penalty shootout}'] != 'Injured', :]
ground_truths = ground_truths.dropna(how='all').reset_index()


# In[ ]:


df = pd.read_csv('./data/real/TwitterWorldCup2014/Twitter_WorldCup_2014_resolved.txt', names=['date', 'entity1', 'entity2'], sep=' ')


# In[ ]:


df['date'] = pd.to_datetime(df['date'], format='%m:%d:%Y:%H:%M:%S')


# In[ ]:


count_od = df.groupby([pd.Grouper(key='date', freq='1h'), 'entity1', 'entity2']).size().reset_index().rename(columns={0: 'count'})


# In[ ]:


terms_array = sorted(count_od['date'].unique())


# In[ ]:


entities_array = pd.concat([df['entity1'], df['entity2']]).unique()


# In[ ]:


infile_entities_dict_name2id = 'data/real/TwitterWorldCup2014/entities_dict_id2name.pkl'
infile_entities_dict_id2name = 'data/real/TwitterWorldCup2014/entities_dict_name2id.pkl'

if os.path.exists(infile_entities_dict_name2id) & os.path.exists(infile_entities_dict_id2name):
    with open(infile_entities_dict_name2id, 'rb') as f:
        entities_dict_name2id = pickle.load(f)
    with open(infile_entities_dict_id2name, 'rb') as f:
        entities_dict_id2name = pickle.load(f)
else:
    entities_dict_name2id = {v: i for i, v in enumerate(entities_array)}
    entities_dict_id2name = {i: v for i, v in enumerate(entities_array)}
    
    with open(infile_entities_dict_name2id, 'wb') as f:
        pickle.dump(entities_dict_name2id, f)
    with open(infile_entities_dict_id2name, 'wb') as f:
        pickle.dump(entities_dict_id2name, f)


# In[ ]:


count_od['entity1'] = [entities_dict_name2id[o] for o in count_od['entity1']]
count_od['entity2'] = [entities_dict_name2id[o] for o in count_od['entity2']]


# In[ ]:


count_od = count_od.loc[(count_od['date'] >= pd.to_datetime('2014-06-01 00:00:00')) & (count_od['date'] <= pd.to_datetime('2014-07-15 23:00:00')), :]


# In[ ]:


with open('data/real/TwitterWorldCup2014/count_od_by_week.pkl', 'wb') as f:
          pickle.dump(count_od, f)


# In[ ]:




