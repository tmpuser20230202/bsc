import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.reset_defaults()
sns.set_context(context='talk', font_scale=0.8)
cmap = plt.get_cmap("tab10")

import scipy.stats as stats
import scipy.sparse as sparse
import scipy.special as special

from functools import partial

import jax
from jax import random, lax
import jax.numpy as jnp
import numpyro

import graph_tool.all as gt

import arviz as az

import networkx as nx   # Version 1.9.1

import cartopy.crs as ccrs

from importlib import reload
import os
import pickle
import joblib
import tqdm


import bsc
import generate_data
reload(bsc)
reload(generate_data)
from bsc import BSC, calc_paracomp_bern_with_prior_beta
from generate_data import DataGeneratorSBM

from cython_normterm_discrete import create_fun_with_mem

def calc_codelength_integer(k):
    codelength = 2.865
    k = np.log(k)
    while k > 0:
        codelength += k
        k = np.log(k)
    
    return codelength


outdir = 'output/sbm2023/synthetic'
if not os.path.exists(outdir):
    os.mkdir(outdir)

# parameters 
K = 3
N = 100
a = 1.0
b = 1.0
du = 0.05
seed = 0

T = 30
T1 = 10
T2 = 20

k_list = [2, 3, 4, 5]

λ = 2.0

gen = DataGeneratorSBM(T)

normterm = create_fun_with_mem()

n_trial = 5
codelen_x = []
codelen_y = []
codelen_z = []

normterm_y = [np.log(np.sum([
                      np.exp(special.loggamma(k**2+1) - 
                             special.loggamma(n_plus+1) - 
                             special.loggamma(k**2-n_plus+1)) *
                      (((n_plus + (a-1))/(k**2+a+b+λ-2))**(n_plus + (a-1)) *
                       ((b-1+λ + k**2-n_plus)/(k**2+a+b+λ-2))**(k**2-n_plus+b-1+λ))
                      for n_plus in range(k**2+1)])) for k in k_list]

for n in range(n_trial):
    X_list, Z_list, eta_former, eta_latter = gen.generate(K, N, T1, T2, a, b, du=0.1, ratio=0.05, seed=n)
    
    g_list_n = []
    vp_list_n = []
    
    codelen_x_n = []
    codelen_y_n = []
    codelen_z_n = []
    
    
    for t in tqdm.tqdm(range(T)):
        g = gt.Graph(directed=True)
        indices_i, indices_j = np.nonzero(X_list[t])
        indices = np.hstack((indices_i.reshape(-1, 1), indices_j.reshape(-1, 1)))
        vp = g.add_edge_list(indices, hashed=True)
        
        g_list_n.append(g)
        vp_list_n.append(vp)
        
        _, counts = np.unique(np.argmax(Z_list[t], axis=1), return_counts=True)
        
        state = gt.minimize_blockmodel_dl(g, deg_corr=False)
        
        codelen_x_t = []
        codelen_y_t = []
        codelen_z_t = []
        codelen_k_t = []
        
        order_codelen_x_t_k_list = []
        kdel_list = []
        for i, k in enumerate(k_list):
            state_k = gt.minimize_blockmodel_dl(g, deg_corr=False, B_min=k, B_max=k)
        
            blocks = list(state_k.get_blocks())
            _, counts = np.unique(blocks, return_counts=True)
        
            codelen_z_t_k = np.sum(-counts * np.log(counts/np.sum(counts))) + np.log(normterm.evaluate(N, k))
            codelen_z_t.append(codelen_z_t_k)
            
            n_all = counts.reshape(-1, 1) * counts.reshape(1, -1)
            n_p = (state_k.get_matrix().toarray()).astype(int)
            n_m = n_all - n_p
            
            codelen_x_t_k = np.where(n_all == 0, 0.0, n_all * np.log(n_all)) - \
                            np.where(n_p == 0, 0.0, n_p * np.log(n_p)) - \
                            np.where(n_m == 0, 0.0, n_m * np.log(n_m)) + \
                            np.log([[normterm.evaluate(n_all[i, j], 2) #if n_all[i, j] >= 2 else 0.0
                                       for j in range(k)] for i in range(k)])

            codelen_x_t_k_noprob = np.log(n_all+1) + special.loggamma(n_all+1) - special.loggamma(n_p+1) - special.loggamma(n_all-n_p+1)
            
            order_codelen_x_t_k = np.dstack(np.unravel_index(np.argsort(codelen_x_t_k_noprob.ravel() -  codelen_x_t_k.ravel()), (k, k)))[0]
            
            order_codelen_x_t_k_list.append(order_codelen_x_t_k)
            
            max_reduction = np.sum(codelen_x_t_k_noprob.ravel() < codelen_x_t_k.ravel() )

            k_del = 0
            codelen_total_del_prev = None
            codelen_x_del_prev = None
            codelen_y_del_prev = None
            while k_del <= max_reduction:
                rho_hat = (k**2 - k_del + a -1.0)/(k**2 + a + b + λ - 2)
                codelen_y_del_curr = -(k**2 - k_del + a -1.0) * np.where(k**2 - k_del + a -1.0 == 0.0, 
                                                                     0.0, 
                                                                     np.log(rho_hat)) \
                                 -(k_del + b + λ - 1.0) * np.where(k_del + b + λ - 1.0 == 0.0,
                                                                   0.0,
                                                                   np.log(1.0 - rho_hat)) + \
                                 normterm_y[i] #+ calc_codelength_integer(k**2 - k_del)
                codelen_x_each = codelen_x_t_k.copy()
                for ind in range(k_del):
                    indices = order_codelen_x_t_k[ind]
                    codelen_x_each[indices[0], indices[1]] = codelen_x_t_k_noprob[indices[0], indices[1]]
            
                codelen_x_del_curr = np.sum(codelen_x_each)
                codelen_total_del_curr = codelen_x_del_curr + codelen_y_del_curr

                
                if codelen_total_del_prev:
                    if codelen_total_del_curr >= codelen_total_del_prev:
                        break
                
                codelen_total_del_prev = codelen_total_del_curr
                codelen_x_del_prev = codelen_x_del_curr
                codelen_y_del_prev = codelen_y_del_curr
                
                k_del += 1
            
            codelen_x_t.append(codelen_x_del_prev)
            codelen_y_t.append(codelen_y_del_prev)
            codelen_k_t.append(calc_codelength_integer(k))
            
            kdel_list.append(k_del)
        
        codelen_x_n.append(codelen_x_t)
        codelen_y_n.append(codelen_y_t)
        codelen_z_n.append(codelen_z_t)
    
    codelen_x.append(codelen_x_n)
    codelen_y.append(codelen_y_n)
    codelen_z.append(codelen_z_n)

codelen_x_z = np.array(codelen_x) + np.array(codelen_z)

