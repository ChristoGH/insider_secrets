# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 13:40:18 2016

@author: Christiaan
"""

nedbankPal = ['#FAAE2C', '#E3DBD1', '#717071', '#AD8F6A', '#515051', '#313031', '#FFFFFF']

#%%
catvecs_clust = []
numvecs_clust = []
catcols = ['Race', 'Language', 'Gender', 'MaritalStatus', 'Province','Seg_l3_str',
           'MainBank', ]
numcols = ['NII', 'NIR',
       'Debits', 'Credits',
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
for y in sorted(list(df_clustered.clusternum.unique())):  
    temp = df_clustered[df_clustered['clusternum']==y]    
    catvecs_clust.append([temp[x].value_counts()/len(temp) for x in catcols])
    numbinned_clust = [pd.cut(temp[x],bins=sorted(list(set(df_clustered[x].quantile([0,.2,.4,.6,.8,1])))),
                  include_lowest=True) for x in numcols]                  
    numbinned_clust = pd.concat(numbinned_clust,axis=1)    
    numvecs_clust.append([numbinned_clust[x].value_counts()/len(temp) for x in numcols])
    
catvecs = [df_clustered[x].value_counts()/len(df_clustered) for x in catcols]
numvecs = [df_clustered[x].value_counts()/len(df_clustered) for x in numcols]
#%% Compare Specific variables numeric
col = 'NII'
ind = numcols.index(col)
x_ind = range(len(numvecs[ind]))
width = 0.15
fig, ax = plt.subplots()
#plt.plot(catcols,catdists)
#plt.plot(catcols,catdists2)
ax.bar([x-1.5*width for x in x_ind], numvecs[ind].sort_index(), width, color='#000000',label='Population')
ax.bar([x-0.5*width for x in x_ind], numvecs_clust[0][ind].sort_index(), width, color=nedbankPal[0],label='Cluster 0')
ax.bar([x+0.5*width for x in x_ind], numvecs_clust[1][ind].sort_index(), width, color=nedbankPal[1],label='Cluster 1')
ax.bar([x+1.5*width for x in x_ind], numvecs_clust[2][ind].sort_index(), width, color=nedbankPal[2],label='Cluster 2')
ax.bar([x+2.5*width for x in x_ind], numvecs_clust[3][ind].sort_index(), width, color=nedbankPal[3],label='Cluster 3')
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
ax.bar([x-0.5*width for x in x_ind], catvecs_clust[0][ind].sort_index(), width, color=nedbankPal[0],label='Cluster 0')
ax.bar([x+0.5*width for x in x_ind], catvecs_clust[1][ind].sort_index(), width, color=nedbankPal[1],label='Cluster 1')
ax.bar([x+1.5*width for x in x_ind], catvecs_clust[2][ind].sort_index(), width, color=nedbankPal[2],label='Cluster 2')
ax.bar([x+2.5*width for x in x_ind], catvecs_clust[3][ind].sort_index(), width, color=nedbankPal[3],label='Cluster 3')
ax.set_xticks(x_ind)
ax.set_title(col)
ax.set_xticklabels(catvecs[ind].sort_index().index.values, rotation='vertical')
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.show()
    