import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys
import glob
import tqdm
import pickle


infile_all = 'data/real/FLIGHTLIST_20190901_20200531/df_FLIGHTLIST_all.pkl'
if not os.path.exists(infile_all):
    infiles = sorted(glob.glob('data/real/FLIGHTLIST_20190901_20200531/*.csv.gz'))
    df_list = []
    for infile in tqdm.tqdm(infiles):
        print(infile)
        df = pd.read_csv(infile)
        df_list.append(df)
    
    df_concat = pd.concat(df_list, axis=0, ignore_index=True)
    
    with open(infile_all, 'wb') as f:
        pickle.dump(df_concat, f, protocol=4)
else:
    with open(infile_all, 'rb') as f:
        df_concat = pickle.load(f)

with open(infile_all, 'wb') as f:
    pickle.dump(df_concat, f, protocol=4)

df_concat['day'] = pd.to_datetime(df_concat['day'])
df_concat_for_grouping = df_concat[['day', 'origin', 'destination']]

count_by_W_O_D = df_concat_for_grouping.groupby([
                     pd.Grouper(key='day', freq='1W'), 
                     'origin', 'destination']).size()
count_by_W_O_D = count_by_W_O_D.reset_index()
count_by_W_O_D.rename({0: 'count'}, axis=1, inplace=True)
count_by_W_O_D['day'] = count_by_W_O_D['day'].dt.date

with open('data/real/FLIGHTLIST_20190901_20200531/week/count_od_by_week.pkl', 'wb') as f:
    pickle.dump(count_by_W_O_D, f)

count_by_W_O_D_20190901_20200531 = count_by_W_O_D.loc[
    (count_by_W_O_D['day'] >= pd.to_datetime('2019-09-01')) & 
    (count_by_W_O_D['day'] <= pd.to_datetime('2020-05-31')), :]

airports_array = np.unique(np.hstack((count_by_W_O_D_20190901_20200531['origin'].unique(), 
                                      count_by_W_O_D_20190901_20200531['destination'].unique())))

infile_airports_dict_name2id = 'data/real/FLIGHTLIST_20190901_20200531/airports_dict_name2id.pkl'
infile_airports_dict_id2name = 'data/real/FLIGHTLIST_20190901_20200531/airports_dict_id2name.pkl'

if os.path.exists(infile_airports_dict_name2id) & os.path.exists(infile_airports_dict_id2name):
    with open(infile_airports_dict_name2id, 'rb') as f:
        airports_dict_name2id = pickle.load(f)
    with open(infile_airports_dict_id2name, 'rb') as f:
        airports_dict_id2name = pickle.load(f)
else:
    airports_dict_name2id = {v: i for i, v in enumerate(airports_array)}
    airports_dict_id2name = {i: v for i, v in enumerate(airports_array)}
    
    with open(infile_airports_dict_name2id, 'wb') as f:
        pickle.dump(airports_dict_name2id, f)
    with open(infile_airports_dict_id2name, 'wb') as f:
        pickle.dump(airports_dict_id2name, f)

count_by_W_O_D_20190901_20200531['origin'] = [airports_dict_name2id[o] for o in count_by_W_O_D_20190901_20200531['origin']]
count_by_W_O_D_20190901_20200531['destination'] = [airports_dict_name2id[o] for o in count_by_W_O_D_20190901_20200531['destination']]
count_by_W_O_D_20190901_20200531.reset_index(drop=True, inplace=True)

weeks_array = count_by_W_O_D_20190901_20200531['day'].unique()
for week in weeks_array:
    print(week)
    df_week = count_by_W_O_D_20190901_20200531.loc[count_by_W_O_D_20190901_20200531.loc[:, 'day'] == week, ['origin', 'destination', 'count']]
    df_week.to_csv(os.path.join('data/real/FLIGHTLIST_20190901_20200531/week', week.strftime('%Y-%m-%d') + '.csv'), 
                   index=None, header=None)

