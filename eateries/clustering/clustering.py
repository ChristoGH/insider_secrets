# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 15:04:44 2016

@author: Christiaan
"""

#%%
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.spatial.distance import cdist, pdist
from mpl_toolkits.mplot3d import Axes3D
#%%
from multiprocessing.dummy import Pool as ThreadPool 
#%%
df_analysis_full = pd.read_hdf('eateries/cluster_data',key='df_scaled')
df_analysis = df_analysis_full.sample(100000)
#%%
# determine optimal amount of clusters
#set seed for repeatable results
#seed 11 k=15 clusters silhouette score 0.24089004302045561
#seed 1 k=2 clusters silh sc 0.195950227414
#seed 1234 k=20 0.199782989696
seed = 0
# set range of amount of clusters
K = range(2,11)
km = [KMeans(init='k-means++', n_clusters=k, n_init = 50, random_state=seed,
             verbose=1, n_jobs=5).fit(df_analysis) for k in K]
centroids = [k.cluster_centers_ for k in km]
D_k = [cdist(df_analysis, cent, 'euclidean') for cent in centroids]
cIdx = [np.argmin(D,axis=1) for D in D_k]
dist = [np.min(D,axis=1) for D in D_k]
# metric one
avgWithinSS = [sum(d)/df_analysis.shape[0] for d in dist]
#%%
# Total with-in sum of square - two
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(df_analysis)**2)/df_analysis.shape[0]
bss = tss-wcss
#%%
# three - best in most cases - silhouette score
# this can take a while :/
#pool = ThreadPool(5)
#sillybilly = pool.starmap(silhouette_score, zip(itertools.repeat(df_scaled), [k.labels_ for k in km],itertools.repeat('euclidean')))
#pool.close()
#pool.join()
sillybilly=[]
i=0
for k in km:
    sillybilly.append(silhouette_score(df_analysis, k.labels_, metric='euclidean', sample_size=10000, n_jobs=5))
    print(sillybilly[i])
    i=i+1
#%%

sillybilly=[]
sillybilly.append(silhouette_score(df_scaled, km[0].labels_, metric='euclidean', n_jobs=2))
#%%

silhouette_samples(df_scaled,km[3].labels_)

#%%
# Plot one
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, avgWithinSS, 'b*-')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
plt.title('Elbow for KMeans clustering')
#%%
#Plot two
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, bss/tss*100, 'b*-')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Percentage of variance explained')
plt.title('Elbow for KMeans clustering')
#%%
#Plot three
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K,sillybilly, 'b*-')
plt.grid(True)
plt.ylabel("Silhouette")
plt.xlabel("k")
plt.title("Silhouette for K-means cell's behaviour")
#%%
#PCA based initialization
pca1 = PCA()
pca1.fit(df_analysis)

plt.figure(1, figsize=(16, 12))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca1.explained_variance_[:20], linewidth=2)
plt.axis('tight')
plt.grid(True)
plt.xlabel('n_components')
plt.ylabel('explained_variance_')

#%%
# Do clustering - set seed and n_clusters based on conclusion above
seed = 0
kmeans_clust = KMeans(init='k-means++', n_clusters=4, n_init=100,random_state=seed,verbose=3,n_jobs=5)

kmeans_clust.fit(df_analysis_full.sample(400000))

#%%
# Write clusters back to dataset
df_clustered = pd.read_hdf('eateries/cluster_data',key='df_raw')
df_clustered['cluster'] = kmeans_clust.predict(df_analysis_full)
df_clustered.to_hdf('eateries/cluster_data',key='df_result_with_clusters_v1',complevel=9)
#%% 
# Plot clusters 2D first pc vs second
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(df_analysis)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=kmeans_clust.predict(df_analysis))
plt.show()
#%%
# Plot clusters 3D first pc vs second vs third
pca_3 = PCA(3)
plot_columns = pca_3.fit_transform(df_analysis)
fig = plt.figure(1,figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(plot_columns[:,0], plot_columns[:,1],plot_columns[:,2], c=kmeans_clust.predict(df_analysis))
plt.show()
#%%
df_centroids = pd.DataFrame(kmeans_clust.cluster_centers_,columns = df_analysis.columns)
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
ss = joblib.load('eateries/standard_scaler.joblib')
df_centroids = pd.DataFrame(ss.inverse_transform(df_centroids), columns = df_analysis.columns)
#%%
for idx, val in df_clustered.cluster.value_counts().iteritems():
    df_centroids.loc[idx, 'observations'] = val
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

#%%
catcols = ['Race', 'Language', 'Gender', 'MaritalStatus', 'Province','Seg_l3_str']
mode_dict = {}
for cluster in df_clustered.cluster.unique():
    mode_dict[cluster] = df_clustered.loc[df_clustered.cluster == cluster,catcols].mode()
df_modes = pd.concat(mode_dict).T
df_modes.columns = df_modes.columns.droplevel(1)
#%%
df_centroids = df_centroids.merge(df_modes.T, left_index=True, right_index=True,suffixes=['','_cat'])
df_centroids.to_excel('clustering_centroids_v1.xlsx')
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
    f, axarr = plt.subplots(5, sharex=True, sharey=False, figsize=(12,48))

    f.suptitle(var)
    filt_val = [True]*len(df_clustered)
    if '_amt' in var:
        filt_val = (df_clustered[var]>-500)
    if '_cnt' in var:
        filt_val = (df_clustered[var]<25)
    col_names = ['grass_green', 'teal_green', 'apple_green', 'orange', 'orange_yellow']
    df_clustered.loc[filt_val,var].hist(bins=100, 
                     color=get_nedbank_colour('nedbank_green'), label='population', ax=axarr[0])
    for cluster in df_clustered.cluster.unique():
        hist_color = get_nedbank_colour(col_names[cluster])
        df_clustered.loc[filt_val&(df_clustered.cluster == cluster),var].hist(bins=100, 
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
    plt.close('all')
