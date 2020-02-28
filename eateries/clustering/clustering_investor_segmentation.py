# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 15:04:44 2016

@author: Christiaan
"""

#%%
import pandas as pd
import numpy as np
import pyodbc
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.spatial.distance import cdist, pdist
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering as agc

#%%
country = 'za'

var_filt = ['age',
            'GrossSalary',
            'RiskScore',
            'days_since_last_loan',
            'CreditCheckPrevEnqTotal',
            'SalaryDateDay',
            'MotivationType',
            'BankRefId',
            'HomeOwnerType',
            'Dependents',
            'Title',
            'MaritalStatus',
            'GenderType']
#%% When using previous training data

df = learn_data[var_filt].copy()

#%%
#Acquire data and set index


SQL = '''
        SELECT a.ClientRefId,
               AVG(CASE
                       WHEN LoanSubState IN(8, 9)
                       THEN 1
                       WHEN LoanSubState IN(7, 11, 13)
                       THEN 0
                       ELSE 0.5
                   END) AS lnStatMetric,
               SUM([PresentValue]) AS totalLoanAmnt,
               COUNT(DISTINCT a.RefId) AS loanCnt,
               COALESCE(c.Name, 'Unknown') AS gender,
               COALESCE(d.Name, 'Unknown') AS maritalStatus,
               COALESCE(e.Name, 'Unknown') AS homelang,        
               2016 - yr AS age,
               aff.[GrossSalary] AS salary
        FROM [dbo].[Loans] a
             LEFT JOIN
        (
            SELECT *,
                   RANK() OVER(PARTITION BY ClientRefId ORDER BY PsmAffordabilityRequestDate,
                                                                 RefId DESC) AS ranknum
            FROM [dbo].[AffordabilityDetails]
        ) aff ON aff.[ClientRefId] = a.ClientRefId
                 AND aff.ranknum = 1 
             INNER JOIN
        (
            SELECT RefId,
                   YEAR(DateOfBirth) AS yr
            FROM [dbo].[Clients]
        ) clnt ON clnt.RefId = a.ClientRefId
             LEFT JOIN [dbo].[PersonalDetails] b ON b.ClientRefId = a.ClientRefId
             LEFT JOIN
        (
            SELECT *
            FROM [Atlas].[dbo].[EnterpriseFieldTypes]
            WHERE [Type] = 6
        ) c ON c.[Key] = b.GenderType
             LEFT JOIN
        (
            SELECT *
            FROM [Atlas].[dbo].[EnterpriseFieldTypes]
            WHERE [Type] = 7
        ) d ON d.[Key] = b.MaritalStatus
             LEFT JOIN
        (
            SELECT *
            FROM [Atlas].[dbo].[EnterpriseFieldTypes]
            WHERE [Type] = 2
        ) e ON e.[Key] = b.LanguageType
        WHERE LoanSubState IN(7, 8, 9, 10, 11, 13)
             AND [LoanEffective] > '2015-07-01'
             AND [LoanEffective] < '2016-07-01'
        GROUP BY a.ClientRefId,
                 c.Name,
                 d.Name,
                 e.Name,
                 yr,
                 aff.GrossSalary
        ORDER BY ClientRefID;
        '''
cnxn = pyodbc.connect(driver='{SQL Server}', server='197.96.22.2,5595',\
                          database='{0}_FinCloud'.format(country), UID='AiUser', PWD='AiUser.Psibyl11', autocommit=True)
                          
df = pd.read_sql_query(SQL, cnxn, index_col='ClientRefId')

cnxn.close()

#%% merge region info if so desired - get this from city_pc_to_region.py
df = df.merge(rgn,left_index=True,right_index=True,how='left')
df.pr_nm2.fillna('unknown', inplace=True)
#%%
#get category columns correct

df['gender']=df['gender'].astype('category')
df['maritalStatus']=df['maritalStatus'].astype('category')
del df['homelang']

#%%
##Remove NULLs / NANS /Whatever else
#null_rows = pd.isnull(df).any(1).nonzero()[0]
#df.loc[df.index[null_rows]]
#df_nonan= df.dropna()
##constant cols
#df = df.loc[:,df.apply(pd.Series.nunique) != 1]
df_nonan = df.fillna(-1)

#%%
# get rid of unrealistic outliers
df_nonan.hist(figsize = (12,9))

max(df_nonan['age'])
len(df_nonan[df_nonan['age']>110])
df_nonan = df_nonan[df_nonan['age']<=110]
max(df_nonan['GrossSalary'])
len(df_nonan[df_nonan['GrossSalary']>10000000])
df_nonan = df_nonan[df_nonan['GrossSalary']<=10000000]
#%% Even binning if required
numcols = list(df_nonan.select_dtypes(exclude=['category']).columns)
numcols.remove('age')
cuts = [y/10 for y in range(0,11,1)]
numbinned = [pd.cut(df_nonan[x],bins=sorted(list(set(df_nonan[x].quantile(cuts)))),
                  include_lowest=True) for x in numcols]                  
numbinned = pd.concat(numbinned,axis=1)

for x in numcols:
    del df_nonan[x]

df_nonan = df_nonan.merge(numbinned, left_index=True,right_index=True)

#%%
#Convert factors to binary columns
dfnum=df_nonan.copy()
dfnum.loc[:,dfnum.dtypes == bool] = dfnum.loc[:,dfnum.dtypes == bool].astype(int)
dfnum = pd.get_dummies(dfnum)
#dfnum.to_csv('spain_clust.csv')
#dfnum.drop('PostalAddressSameAsPhysical_False',axis=1, inplace=True)

#%%
#Scale to z-scores
df_scaled = StandardScaler().fit_transform(dfnum)

#%%
# determine optimal amount of clusters
#set seed for repeatable results
#seed 11 k=15 clusters silhouette score 0.24089004302045561
#seed 1 k=2 clusters silh sc 0.195950227414
#seed 1234 k=20 0.199782989696
seed = 0
# set range of amount of clusters
K = range(2,11)
km = [KMeans(init='k-means++', n_clusters=k, n_init = 100, random_state=seed).fit(df_scaled) for k in K]
centroids = [k.cluster_centers_ for k in km]
D_k = [cdist(df_scaled, cent, 'euclidean') for cent in centroids]
cIdx = [np.argmin(D,axis=1) for D in D_k]
dist = [np.min(D,axis=1) for D in D_k]
# metric one
avgWithinSS = [sum(d)/df_scaled.shape[0] for d in dist]
#%%
# Total with-in sum of square - two
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(df_scaled)**2)/df_scaled.shape[0]
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
    sillybilly.append(silhouette_score(df_scaled, k.labels_, metric='cityblock', n_jobs=2, sample_size=20000)) #, sample_size=20000
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
pca1.fit(df_scaled)

plt.figure(1, figsize=(16, 12))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca1.explained_variance_[:15], linewidth=2)
plt.axis('tight')
plt.grid(True)
plt.xlabel('n_components')
plt.ylabel('explained_variance_')

#%%
# Choose components above and then do the following
sillybilly=[]
K = [2,3,4,5,6,7,8,9,10]
for n_comp in K:
    pca = PCA(n_components=n_comp).fit(df_scaled)
    kmeans_clust = KMeans(init=pca.components_, n_clusters=n_comp, n_init=1).fit(df_scaled)
    sillybilly.append(silhouette_score(df_scaled, kmeans_clust.labels_, metric='cityblock',n_jobs=5, sample_size=20000)) #sample_size=20000,  
    print(sillybilly)


#%%
# Do clustering - set seed and n_clusters based on conclusion above
seed = 0
kmeans_clust = KMeans(init='k-means++', n_clusters=3, n_init=100,random_state=seed,verbose=3)

kmeans_clust.fit(df_scaled)

n_comp = 4
pca = PCA(n_components=n_comp).fit(df_scaled)
kmeans_clust = KMeans(init=pca.components_, n_clusters=n_comp, n_init=1,verbose=3).fit(df_scaled)

#%%

silhouette_score(df_scaled, kmeans_clust.labels_, metric='cityblock', n_jobs=5, sample_size=20000)

#%%
# Write clusters back to dataset
df_clust = df_nonan.copy()
df_clust['cluster'] = kmeans_clust.labels_
df_clust.cluster.value_counts()
#%% 
# Plot clusters 2D first pc vs second
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(df_scaled)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=kmeans_clust.labels_)
plt.show()
#%%
# Plot clusters 3D first pc vs second vs third
pca_3 = PCA(3)
plot_columns = pca_3.fit_transform(df_scaled)
fig = plt.figure(1,figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(plot_columns[:,0], plot_columns[:,1],plot_columns[:,2], c=kmeans_clust.labels_)
plt.show()

#%%
popmean = df_clust.mean().values[:5]
df_clust.groupby(['cluster']).apply(lambda x: abs(x.mean()[:5]-popmean))
#%% binning
df_out = df_clust.copy()
df_clust.age.hist()
df_out['age'] = pd.cut(df_clust['age'],bins=[0,20,30,40,50,60,70,80,90,100])

df_clust.lnStatMetric.hist()
df_out['lnStatMetric'] = pd.cut(df_clust['lnStatMetric'],bins=10)

df_clust.totalLoanAmnt.max()
df_clust.totalLoanAmnt.hist(bins=(list(range(0,80001,8000))+[df_clust.totalLoanAmnt.max()]))
df_out['totalLoanAmnt'] = pd.cut(df_clust['totalLoanAmnt'],bins=(list(range(0,80001,8000))+[df_clust.totalLoanAmnt.max()]))

df_clust.loanCnt.max()
df_clust.loanCnt.hist(bins=(list(range(0,21,2))+[df_clust.loanCnt.max()]))
df_out['loanCnt'] = pd.cut(df_clust['loanCnt'],bins=(list(range(0,21,2))+[df_clust.loanCnt.max()]))

df_clust.salary.max()
df_clust.salary.hist(bins=(list(range(0,100001,10000))))
df_out['salary'] = pd.cut(df_clust['salary'],bins=(list(range(0,100001,10000))+[df_clust.salary.max()]))

#%% Hierarchical dataset for powerbi

df_dict ={}
y=0
for x in list(df_out.columns - ['cluster']):    
    df_dict[x] = df_out.groupby(['cluster',x]).count().iloc[:,0]
    df_dict[x].fillna(0,inplace=True)
    df_dict[x]=pd.DataFrame(df_dict[x])
    df_dict[x].columns = ['value']
#    df_dict[x]['var']=x
#    df_dict[x].set_index('var', append=True, inplace=True)
#    df_dict[x] = df_dict[x].reorder_levels(['var', 'cluster', x])
    y = y+1
    print(y/len(list(df_out.columns - ['cluster'])))

df_h = pd.concat(df_dict)

df_pop = {}
for x in list(df_out.columns - ['cluster']):    
    df_pop[x] = df_out.groupby([x]).count().iloc[:,0]
    df_pop[x].fillna(0, inplace=True)
    df_pop[x]=pd.DataFrame(df_pop[x])
    df_pop[x].columns = ['value']    
    df_pop[x]['cluster']=-1
    df_pop[x].set_index('cluster', append=True, inplace=True)
    df_pop[x] = df_pop[x].reorder_levels(['cluster', x])
    
df_h_pop = pd.concat(df_pop)

df_h = pd.concat([df_h,df_h_pop])

#%%

df_h.to_excel('data/za_finyr_201507_201606_v1.xlsx')