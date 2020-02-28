# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 13:40:18 2016

@author: Christiaan
"""

getbucksPal = ['#FAAE2C', '#E3DBD1', '#717071', '#AD8F6A', '#515051', '#313031', '#FFFFFF']

N = len(df_clust.index)
#%% Population Categorical Vectors

catcols = list(df_clust.select_dtypes(include=['category']).columns)
for x in catcols:
    df_clust[x] = df_clust[x].astype('category')
catvecs = [df_clust[x].value_counts()/N for x in catcols]

#%% Population numerical Vectors
numcols = list(df_clust.select_dtypes(exclude=['category']).columns)
numcols.remove('cluster')
numbinned = [pd.cut(df_clust[x],bins=sorted(list(set(df_clust[x].quantile([0,.2,.4,.6,.8,1])))),
                  include_lowest=True) for x in numcols]                  
numbinned = pd.concat(numbinned,axis=1)
numvecs = [numbinned[x].value_counts()/N for x in numcols]


#%%
catvecs_clust = []
numvecs_clust = []
for y in sorted(list(df_clust.cluster.unique())):  
    temp = df_clust[df_clust['cluster']==y]    
    catvecs_clust.append([temp[x].value_counts()/len(temp) for x in catcols])
    numbinned_clust = [pd.cut(temp[x],bins=sorted(list(set(df_clust[x].quantile([0,.2,.4,.6,.8,1])))),
                  include_lowest=True) for x in numcols]                  
    numbinned_clust = pd.concat(numbinned_clust,axis=1)    
    numvecs_clust.append([numbinned_clust[x].value_counts()/len(temp) for x in numcols])
    


#%% Compare Specific variables numeric
col = 'lnStatMetric'
ind = numcols.index(col)
x_ind = range(len(numvecs[ind]))
width = 0.15
fig, ax = plt.subplots()
#plt.plot(catcols,catdists)
#plt.plot(catcols,catdists2)
ax.bar([x-1.5*width for x in x_ind], numvecs[ind].sort_index(), width, color='#000000',label='Population')
ax.bar([x-0.5*width for x in x_ind], numvecs_clust[0][ind].sort_index(), width, color=getbucksPal[0],label='Cluster 0')
ax.bar([x+0.5*width for x in x_ind], numvecs_clust[1][ind].sort_index(), width, color=getbucksPal[1],label='Cluster 1')
ax.bar([x+1.5*width for x in x_ind], numvecs_clust[2][ind].sort_index(), width, color=getbucksPal[2],label='Cluster 2')
ax.bar([x+2.5*width for x in x_ind], numvecs_clust[3][ind].sort_index(), width, color=getbucksPal[3],label='Cluster 3')
ax.set_xticks(x_ind)
ax.set_title(col)
ax.set_xticklabels(numvecs[ind].sort_index().index.values, rotation='vertical')
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.show()
#%% Compare Specific variables categorical
col = 'empsect'
ind = catcols.index(col)
x_ind = range(len(catvecs[ind]))
width = 0.15
fig, ax = plt.subplots()
#plt.plot(catcols,catdists)
ax.bar([x-1.5*width for x in x_ind], catvecs[ind].sort_index(), width, color='#000000',label='Population')
ax.bar([x-0.5*width for x in x_ind], catvecs_clust[0][ind].sort_index(), width, color=getbucksPal[0],label='Cluster 0')
ax.bar([x+0.5*width for x in x_ind], catvecs_clust[1][ind].sort_index(), width, color=getbucksPal[1],label='Cluster 1')
ax.bar([x+1.5*width for x in x_ind], catvecs_clust[2][ind].sort_index(), width, color=getbucksPal[2],label='Cluster 2')
ax.bar([x+2.5*width for x in x_ind], catvecs_clust[3][ind].sort_index(), width, color=getbucksPal[3],label='Cluster 3')
ax.set_xticks(x_ind)
ax.set_title(col)
ax.set_xticklabels(catvecs[ind].sort_index().index.values, rotation='vertical')
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.show()
    