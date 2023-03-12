#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


from bsc import calc_paracomp_bern_with_prior_beta
from cython_normterm_discrete import create_fun_with_mem


# In[3]:


outdir = './output/kdd2023/real/twitter/week'

if not os.path.exists(outdir):
    os.makedirs(outdir)


# In[4]:


def calc_codelength_integer(k):
    codelength = 2.865
    k = np.log(k)
    while k > 0:
        codelength += k
        k = np.log(k)
    
    return codelength


# In[5]:


with open('data/real/TwitterWorldCup2014/count_od_by_week.pkl', 'rb') as f:
    count_od = pickle.load(f)


# In[7]:


count_od


# In[8]:


datetimes_array = sorted(count_od['date'].unique())


# In[9]:


entity_array = np.unique(np.hstack((count_od['entity1'].unique(), count_od['entity2'].unique())))


# In[10]:


g_list = [] 
vp_list = []
for dt in datetimes_array:
    edges = count_od.loc[count_od['date']==dt, ['entity1', 'entity2']].values
    
    g = gt.Graph(directed=False)
    vp = g.add_edge_list(edges, hashed=True)

    g_list.append(g)
    vp_list.append(vp)


# In[11]:


λ = 0.1


# In[12]:


k_list = [55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120]


# In[14]:


states_list = []
for dt, g in tqdm.tqdm(zip(datetimes_array, g_list)):
    print(dt)
    states_list_dt = []
    for k in k_list:
        states = gt.minimize_blockmodel_dl(g, deg_corr=False, B_min=k, B_max=k)
        states_list_dt.append(states)
        
    states_list.append(states_list_dt)


# In[15]:


norm_multinom = create_fun_with_mem()


# In[ ]:


codelen_list = []
codelen_x_list = []
codelen_y_list = []
codelen_z_list = []

for t in tqdm.tqdm(range(len(states_list))):
    g_t = g_list[t]
    X_t = gt.adjacency(g_t).toarray()

    codelen_list_t = []
    codelen_x_list_t = []
    codelen_z_list_t = []

    for j, k in tqdm.tqdm(enumerate(k_list)):
        blocks = list(states_list[t][j].get_blocks())

        _, counts = np.unique(blocks, return_counts=True)
        n_all = counts.reshape(-1, 1) * counts.reshape(1, -1)

        n_blocks = states_list[t][j].B
        n_nodes = states_list[t][j].get_N()
        codelen_z = np.sum(-counts * np.log(counts/np.sum(counts))) + np.log(norm_multinom.evaluate(n_nodes, n_blocks))
        codelen_z_list_t.append(codelen_z)



        codelen_x = 0.0

        n_p = states_list[t][j].get_matrix().toarray()
        n_m = n_all - n_p

        n_all[n_all == 0] = 1
        n_p[n_p == 0] = 1
        n_m[n_m == 0] = 1

        codelen_x = 0.0
        codelen_x += np.sum(n_all * np.log(n_all))
        codelen_x -= np.sum(n_p * np.log(n_p))
        codelen_x -= np.sum(n_m * np.log(n_m))
        codelen_x += np.sum([np.log(norm_multinom.evaluate(n, 2)) if n > 1 else 0.0 for n in n_all.ravel()])

        codelen_x_list_t.append(codelen_x)

        codelen_list_t.append(codelen_x + codelen_z + calc_codelength_integer(k))

    codelen_list.append(codelen_list_t)
    codelen_x_list.append(codelen_x_list_t)
    codelen_z_list.append(codelen_z_list_t)


# In[15]:


codelen_array = np.array(codelen_list)


# In[16]:


indices_one = np.argmin(codelen_array, axis=1)


# In[17]:


scores_both = np.vstack((np.nan*np.ones(codelen_array.shape[1]), codelen_array[:-1, ])) + codelen_array


# In[18]:


indices_both = np.argmin(scores_both, axis=1)


# In[ ]:





# In[19]:


scores = [np.nan]
for i, week in enumerate(weeks_array[1:]):
    score =scores_both[i+1, indices_both[i+1]] - codelen_array[i, indices_one[i]] - codelen_array[i+1, indices_one[i+1]]
    scores.append(score)


# In[ ]:




