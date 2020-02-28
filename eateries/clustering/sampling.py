# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 15:17:44 2016

@author: Christiaan
"""
#%% Load libraries
import pandas as pd
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
#%% Load Data
df_clust=pd.read_excel('data/cntrlGrpza_clustered.xlsx')
df_clust=df_clust.set_index(['ClientRefId'])
#%% Caclulate sample size

Z_a_2 = 2.575   #alpha 0.01     1.96    # alpha 0.05
moe = 0.05      # desired margin of error
p = 0.5         # sample proportion 50% conservative and gives the largest sample size. 

# This can often be determined by using the results from a previous survey, or by running a small pilot study

X = (Z_a_2**2)*p*(1-p)/(moe**2)

N = len(df_clust.index) #Population size

ss = ceil(N*X/(X+N-1)) #this is what we want!

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

#%% ######################## Benchmark SRS ####################################
iter1 = 1000
rsts = np.random.randint(0,100000,iter1)
#%%
for i in range(iter1):
#%% do sampling benchmark SRS
    bench_samp = df_clust.sample(ss,random_state = rsts[i])

#%% Benchmark SRS Categorical Vectors

    bench_catvecs = [bench_samp[x].value_counts()/ss for x in catcols]

#%% Benchmark numerical Vectors

    bench_numbinned = [pd.cut(bench_samp[x],bins=sorted(list(set(df_clust[x].quantile([0,.2,.4,.6,.8,1])))),
                  include_lowest=True) for x in numcols]                  
    bench_numbinned = pd.concat(bench_numbinned,axis=1)
    bench_numvecs = [bench_numbinned[x].value_counts()/ss for x in numcols]

#%% Distances benchmark sample from population

    catdists = [np.linalg.norm(bench_catvecs[x]-catvecs[x]) for x in range(len(catvecs))]
    numdists = [np.linalg.norm(bench_numvecs[x]-numvecs[x]) for x in range(len(numvecs))]
    
#%% Get the best one

    if i==0:
        bbs = bench_samp
        bcd = catdists
        bnd = numdists
        bcatvecs = bench_catvecs
        bnumvecs = bench_numvecs
        brsts = rsts[i]
    elif (sum(catdists)+sum(numdists)) < (sum(bcd)+sum(bnd)):
        bbs = bench_samp
        bcd = catdists
        bnd = numdists
        bcatvecs = bench_catvecs
        bnumvecs = bench_numvecs
        brsts = rsts[i]
        
#%% ########################## Smart Sampling #################################
iter1 = 2000
rsts = np.random.randint(0,1000000,iter1)
#%%
for i in range(iter1):
#%% Sample from clusters
    smart_samp = [df_clust[df_clust['cluster']==x].sample(frac=(ss/N),
                  random_state = rsts[i]) for x in df_clust.cluster.unique()]
    smart_samp = pd.concat(smart_samp,axis=0)
    sss = len(smart_samp.index)

#%% SS Categorical Vectors

    smart_catvecs = [smart_samp[x].value_counts()/sss for x in catcols]

#%% Benchmark numerical Vectors

    smart_numbinned = [pd.cut(smart_samp[x],bins=sorted(list(set(df_clust[x].quantile([0,.2,.4,.6,.8,1])))),
                  include_lowest=True) for x in numcols]                  
    smart_numbinned = pd.concat(smart_numbinned,axis=1)
    smart_numvecs = [smart_numbinned[x].value_counts()/sss for x in numcols]

#%% Distances benchmark sample from population

    catdists2 = [np.linalg.norm(smart_catvecs[x]-catvecs[x]) for x in range(len(catvecs))]
    numdists2 = [np.linalg.norm(smart_numvecs[x]-numvecs[x]) for x in range(len(numvecs))]

#%% Get the best one

    if i==0:
        bss = smart_samp
        bcd2 = catdists2
        bnd2 = numdists2
        scatvecs = smart_catvecs
        snumvecs = smart_numvecs
        brsts2 = rsts[i]
    elif (sum(catdists2) < sum(bcd2)) and (sum(numdists2)<sum(bnd2)):
        bss = smart_samp
        bcd2 = catdists2
        bnd2 = numdists2
        scatvecs = smart_catvecs
        snumvecs = smart_numvecs
        brsts2 = rsts[i]
        print(np.array([['Bench',sum(bcd),sum(bnd)],['Smart',sum(bcd2),sum(bnd2)]]))
        
    #if sum(bcd2)<sum(bcd) and sum(bnd2)<sum(bnd): break

#%% Quick Check Bests between bench and smart

np.array([['Bench',sum(bcd),sum(bnd)],['Smart',sum(bcd2),sum(bnd2)]])
        
#%% Plot Results
x_ind = range(len(catcols))
width = 0.15
fig, ax = plt.subplots()
#plt.plot(catcols,catdists)
#plt.plot(catcols,catdists2)
ax.bar(x_ind, bcd, width, color='#000000',label='Benchmark')
ax.bar([x+width for x in x_ind], bcd2, width, color='#FAAE2C',label='Smart Sample')
ax.set_xticks(x_ind)
ax.set_xticklabels(catcols)
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.show()
#%% Plot Results
x_ind = range(len(numcols))
width = 0.15
fig, ax = plt.subplots()
#plt.plot(catcols,catdists)
#plt.plot(catcols,catdists2)
ax.bar(x_ind, bnd, width, color='#000000',label='Benchmark')
ax.bar([x+width for x in x_ind], bnd2, width, color='#FAAE2C',label='Smart Sample')
ax.set_xticks(x_ind)
ax.set_xticklabels(numcols)
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.show()
#%% Compare Specific variables numeric
col = 'ttlLnAmt'
ind = numcols.index(col)
x_ind = range(len(numvecs[ind]))
width = 0.15
fig, ax = plt.subplots()
#plt.plot(catcols,catdists)
#plt.plot(catcols,catdists2)
ax.bar([x-1.5*width for x in x_ind], numvecs[ind].sort_index(), width, color='#000000',label='Population')
ax.bar([x-0.5*width for x in x_ind], bnumvecs[ind].sort_index(), width, color='#666666',label='Benchmark')
ax.bar([x+0.5*width for x in x_ind], snumvecs[ind].sort_index(), width, color='#FAAE2C',label='Smart Sample')
ax.set_xticks(x_ind)
ax.set_title(col)
ax.set_xticklabels(numvecs[ind].sort_index().index.values, rotation='vertical')
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.show()
#%% Compare Specific variables categorical
col = 'maritalStatus'
ind = catcols.index(col)
x_ind = range(len(catvecs[ind]))
width = 0.15
fig, ax = plt.subplots()
#plt.plot(catcols,catdists)
ax.bar([x-1.5*width for x in x_ind], catvecs[ind].sort_index(), width, color='#000000',label='Population')
ax.bar([x-0.5*width for x in x_ind], bcatvecs[ind].sort_index(), width, color='#666666',label='Benchmark')
ax.bar([x+0.5*width for x in x_ind], scatvecs[ind].sort_index(), width, color='#FAAE2C',label='Smart Sample')
ax.set_xticks(x_ind)
ax.set_title(col)
ax.set_xticklabels(catvecs[ind].sort_index().index.values, rotation='vertical')
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.show()