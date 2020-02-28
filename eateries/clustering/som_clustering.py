# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 14:16:26 2017

@author: Christiaan
"""

from sompy.sompy import SOMFactory
import numpy as np
import pandas as pd
#%%
df_cluster = pd.read_hdf('eateries/cluster_data',key='df_numeric')
#%%
# SOM Training
mapsize = [60,60]

som = SOMFactory().build(df_cluster.values, mapsize,normalization = 'var', initialization='pca', component_names=list(df_cluster.columns.values))
som.train(n_job=8, verbose='info', train_rough_len=6, train_finetune_len=100)
topographic_error = som.calculate_topographic_error()
quantization_error = np.mean(som._bmu[1])
print ("Topographic error = %s; Quantization error = %s" % (topographic_error, quantization_error))
#%%
from sompy.visualization.mapview import View2D
#%% View the SOM
view2D  = View2D(60,60,"rand data",text_size=10)
view2D.show(som, col_sz=2, which_dim="all", denormalize=True)
#%%
from sompy.visualization.hitmap import HitMapView
#%% Do clustering on SOM
som.cluster(5)
#setattr(som, 'cluster_labels',df_clust_num.cluster.values)
hits  = HitMapView(60,60,"Clustering",text_size=12)
a=hits.show(som)
#%% Write the SOM labels and clusters back to original dataset
df_clustered = pd.read_hdf('eateries/cluster_data',key='df_raw')
df_clustered['SOM_Labels'] =pd.Series(som.project_data(df_cluster.values),df_cluster.index)
som_cluster_map = pd.Series(getattr(som, 'cluster_labels'))
df_clustered['clusternum'] = df_clustered.SOM_Labels.map(som_cluster_map)

#%% write results out

df_clustered.to_hdf('eateries/cluster_data',key='df_result_with_clusters',complevel=9)
#%%
numcols = ['NII', 'NIR',
       'Debits', 'Credits','MainBank',
       'BURGER KING_amt', 'CHICKEN LICKEN_amt', 'COLCACCHIO_amt',
       'COMPASS GROUP_amt', 'DEBONAIRS_amt', 'DOMINOS PIZZA_amt',
       'DOPPIO ZERO_amt', 'FISH AWAYS_amt', 'FOURNOS_amt', 'KAUAI_amt',
       'KFC_amt', 'KRISPY KREME_amt', 'KUNG FU KITCHEN_amt', 'MCD_amt',
       'MOCHACHOS_amt', 'NANDOS_amt', 'PAPACHINOS_amt', 'PIZZA HUT_amt',
       'ROCOMAMAS_amt', 'SAUSAGE SALOON_amt', 'SIMPLY ASIA_amt', 'SPUR_amt',
       'STEERS_amt', 'WIESENHOF_amt', 'WIMPY_amt', 'BURGER KING_cnt',
       'CHICKEN LICKEN_cnt', 'COLCACCHIO_cnt', 'COMPASS GROUP_cnt',
       'DEBONAIRS_cnt', 'DOMINOS PIZZA_cnt', 'DOPPIO ZERO_cnt',
       'FISH AWAYS_cnt', 'FOURNOS_cnt', 'KAUAI_cnt', 'KFC_cnt',
       'KRISPY KREME_cnt', 'KUNG FU KITCHEN_cnt', 'MCD_cnt', 'MOCHACHOS_cnt',
       'NANDOS_cnt', 'PAPACHINOS_cnt', 'PIZZA HUT_cnt', 'ROCOMAMAS_cnt',
       'SAUSAGE SALOON_cnt', 'SIMPLY ASIA_cnt', 'SPUR_cnt', 'STEERS_cnt',
       'WIESENHOF_cnt', 'WIMPY_cnt', 'client_age']

means_dict = {}
for cluster in df_clustered.clusternum.unique():
    means_dict['cluster_'+str(cluster)] = df_clustered.loc[df_clustered.clusternum == cluster,numcols].fillna(0).mean()
df_centroids = pd.DataFrame(means_dict).T
#%%
for idx, val in df_clustered.clusternum.value_counts().iteritems():
    clust_name = 'cluster_'+str(idx)
    df_centroids.loc[clust_name, 'observations'] = val
#%%
catcols = ['Race', 'Language', 'Gender', 'MaritalStatus', 'Province','Seg_l3_str']
mode_dict = {}
for cluster in df_clustered.clusternum.unique():
    mode_dict['cluster_'+str(cluster)] = df_clustered.loc[df_clustered.clusternum == cluster,catcols].mode()
df_modes = pd.concat(mode_dict).T
df_modes.columns = df_modes.columns.droplevel(1)
#%%
df_centroids = df_centroids.merge(df_modes.T, left_index=True, right_index=True)
df_centroids.to_excel('clustering_centroids.xlsx')
#%%
import matplotlib.pyplot as plt
#%%
def get_nedbank_colour(col_name):

    return {'light_gray':(246/255,246/255,244/255),

            'nedbank_green':(0/255,99/255,65/255),

            'heritage_green':(17/255,87/255,64/255),

            'grass_green':(0,150/255,57/255,1),

            'teal_green':(0,179/255,136/255,1),

            'apple_green':(118/255,183/255,40/255,1),

            'lime_green':(13/255,205/255,0,1),

            'blue_green':(13/255,152/255,186/255,1),

            'orange':(252/255,165/255,0,1),

            'orange_yellow':(255/255,204/255,0,1)

            }.get(col_name)
#%% 
for var in numcols:
    f, axarr = plt.subplots(6, sharex=True, sharey=False, figsize=(12,48))

    f.suptitle(var)
    filt_val = [True]*len(df_clustered)
    if '_amt' in var:
        filt_val = (df_clustered[var]>-500)
    if '_cnt' in var:
        filt_val = (df_clustered[var]<25)
    col_names = ['grass_green', 'teal_green', 'apple_green', 'orange', 'orange_yellow']
    df_clustered.loc[filt_val,var].hist(bins=100, 
                     color=get_nedbank_colour('nedbank_green'), label='population', ax=axarr[0])
    for cluster in df_clustered.clusternum.unique():
        hist_color = get_nedbank_colour(col_names[cluster])
        df_clustered.loc[filt_val&(df_clustered.clusternum == cluster),var].hist(bins=100, 
                        color=hist_color, label='cluster_'+str(cluster), ax=axarr[cluster+1])
    f.subplots_adjust(hspace=0.1)
    # Hide x labels and tick labels for all but bottom plot.
    for enum,ax in enumerate(axarr):
        ax.label_outer()
        if enum>0:
            ax.set_title('cluster_'+str(enum-1))
        else:
            ax.set_title('population')
    f.savefig(var+'_hist.png')
