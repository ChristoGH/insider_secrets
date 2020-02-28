# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 15:09:31 2017

@author: Christiaan
"""

#%%
import pandas as pd
import numpy as np

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

import pyodbc
#%%
cnxn = pyodbc.connect(driver='{SQL Server}', server='10.80.35.19\RELEASEAF',\
                              database='za_FinCloud', UID='AiUser', PWD='AiUser.Psibyl11', autocommit=True) 

from_date = '2017-02-01 00:00:00'
to_date = '2017-02-23 23:59:59'

the_SQL = '''SELECT  SessionId,
        SUM(CASE WHEN ActionId=1 THEN 1 ELSE 0 END) AS LoginSum,
        SUM(CASE WHEN ActionId=2 THEN 1 ELSE 0 END) AS FacebookLoginSum,
        SUM(CASE WHEN ActionId=3 THEN 1 ELSE 0 END) AS LogoutSum,
        SUM(CASE WHEN ActionId=4 THEN 1 ELSE 0 END) AS RegisterSum,
        SUM(CASE WHEN ActionId=5 THEN 1 ELSE 0 END) AS ViewDebitOrdersSum,
        SUM(CASE WHEN ActionId=6 THEN 1 ELSE 0 END) AS ViewProfileQuestionsSum,
        SUM(CASE WHEN ActionId=7 THEN 1 ELSE 0 END) AS AnswerProfileQuestionsSum,
        SUM(CASE WHEN ActionId=8 THEN 1 ELSE 0 END) AS DoesNotQualifySum,
        SUM(CASE WHEN ActionId=9 THEN 1 ELSE 0 END) AS ViewTransactionsSum,
        SUM(CASE WHEN ActionId=10 THEN 1 ELSE 0 END) AS ViewStatementSum,
        SUM(CASE WHEN ActionId=11 THEN 1 ELSE 0 END) AS ViewContractSum,
        SUM(CASE WHEN ActionId=12 THEN 1 ELSE 0 END) AS VerifyInformationSum,
        SUM(CASE WHEN ActionId=13 THEN 1 ELSE 0 END) AS UreditReportSubscribeSum,
        SUM(CASE WHEN ActionId=14 THEN 1 ELSE 0 END) AS CreditReportUnsubscribeSum,
        SUM(CASE WHEN ActionId=15 THEN 1 ELSE 0 END) AS DebtBustersOpenSum,
        SUM(CASE WHEN ActionId=16 THEN 1 ELSE 0 END) AS UpdatePersonalDetailsSum,
        SUM(CASE WHEN ActionId=17 THEN 1 ELSE 0 END) AS UpdateContactDetailsSum,
        SUM(CASE WHEN ActionId=18 THEN 1 ELSE 0 END) AS UpdateEmploymentDetailsSum,
        SUM(CASE WHEN ActionId=19 THEN 1 ELSE 0 END) AS UpdateBankingDetailsSum,
        SUM(CASE WHEN ActionId=20 THEN 1 ELSE 0 END) AS SaveNextOfKinDetailsSum,
        SUM(CASE WHEN ActionId=21 THEN 1 ELSE 0 END) AS UpdateAffordabilityDetailsSum,
        SUM(CASE WHEN ActionId=22 THEN 1 ELSE 0 END) AS ApplyForLoanSum,
        SUM(CASE WHEN ActionId=23 THEN 1 ELSE 0 END) AS AcceptLoanSum,
        SUM(CASE WHEN ActionId=24 THEN 1 ELSE 0 END) AS DeclineLoanSum,
        SUM(CASE WHEN ActionId=25 THEN 1 ELSE 0 END) AS ViewSliderSum,
        SUM(CASE WHEN ActionId=26 THEN 1 ELSE 0 END) AS StartYodleeSum,
        SUM(CASE WHEN ActionId=27 THEN 1 ELSE 0 END) AS SelectBankSum,
        SUM(CASE WHEN ActionId=28 THEN 1 ELSE 0 END) AS LogIntoBankSum,
        SUM(CASE WHEN ActionId=29 THEN 1 ELSE 0 END) AS SelectBankAccountSum,
        SUM(CASE WHEN ActionId=30 THEN 1 ELSE 0 END) AS ConfirmSalarySum,
        SUM(CASE WHEN ActionId=31 THEN 1 ELSE 0 END) AS YodleeCompleteSum,
        SUM(CASE WHEN ActionId=32 THEN 1 ELSE 0 END) AS LinkFacebookSum,
        SUM(CASE WHEN ActionId=33 THEN 1 ELSE 0 END) AS LinkTwitterSum,
        SUM(CASE WHEN ActionId=34 THEN 1 ELSE 0 END) AS WalletApplySum,
        SUM(CASE WHEN ActionId=35 THEN 1 ELSE 0 END) AS ViewPolicyDeclarationSum,
        SUM(CASE WHEN ActionId=36 THEN 1 ELSE 0 END) AS ViewPolicyTermsSum,
        SUM(CASE WHEN ActionId=37 THEN 1 ELSE 0 END) AS ViewPolicyStatutoryNoticeSum,
        SUM(CASE WHEN ActionId=38 THEN 1 ELSE 0 END) AS InstantorStartedSum,
        SUM(CASE WHEN ActionId=39 THEN 1 ELSE 0 END) AS InstantorCompletedSum,
        SUM(CASE WHEN ActionId=40 THEN 1 ELSE 0 END) AS WirecardStartedSum,
        SUM(CASE WHEN ActionId=41 THEN 1 ELSE 0 END) AS WirecardCompletedSum,
        SUM(CASE WHEN ActionId=42 THEN 1 ELSE 0 END) AS TrustlyStartedSum,
        SUM(CASE WHEN ActionId=43 THEN 1 ELSE 0 END) AS TrustlyCompletedSum,
        SUM(CASE WHEN ActionId=44 THEN 1 ELSE 0 END) AS PageLeaveSum,
        SUM(CASE WHEN ActionId=45 THEN 1 ELSE 0 END) AS WidgetLeaveSum,
        SUM(CASE WHEN ActionId=46 THEN 1 ELSE 0 END) AS AffordabilityBypassSum,
        SUM(CASE WHEN ActionId=47 THEN 1 ELSE 0 END) AS UploadedDocumentsSum,
        SUM(CASE WHEN ActionId=48 THEN 1 ELSE 0 END) AS UploadDocumentSum,
        SUM(CASE WHEN ActionId=49 THEN 1 ELSE 0 END) AS RetryAffordabilitySum,
        SUM(CASE WHEN ActionId=50 THEN 1 ELSE 0 END) AS OnePennySelectedSum,
        SUM(CASE WHEN ActionId=51 THEN 1 ELSE 0 END) AS AppResumeSum,
        SUM(CASE WHEN ActionId=52 THEN 1 ELSE 0 END) AS FacebookRegisterSum,
	   SUM(CASE WHEN ActionId=53 THEN 1 ELSE 0 END) AS myWalletViewMyLoans,
	   SUM(CASE WHEN ActionId=54 THEN 1 ELSE 0 END) AS myWalletViewHistory,
	   SUM(CASE WHEN ActionId=55 THEN 1 ELSE 0 END) AS myWalletViewBuy,
	   SUM(CASE WHEN ActionId=56 THEN 1 ELSE 0 END) AS myWalletViewPay,
	   SUM(CASE WHEN ActionId=57 THEN 1 ELSE 0 END) AS myWalletPayoutToBank,
	   SUM(CASE WHEN ActionId=58 THEN 1 ELSE 0 END) AS myWalletPayoutToEwallet,
	   SUM(CASE WHEN ActionId=59 THEN 1 ELSE 0 END) AS myWalletPayoutToFriend,
	   SUM(CASE WHEN ActionId=60 THEN 1 ELSE 0 END) AS myWalletViewHelp,
	   SUM(CASE WHEN ActionId=61 THEN 1 ELSE 0 END) AS myWalletBuyAirtime,
	   SUM(CASE WHEN ActionId=62 THEN 1 ELSE 0 END) AS myWalletBuyElectricity,
	   SUM(CASE WHEN ActionId=63 THEN 1 ELSE 0 END) AS myWalletBuyBill,
	   SUM(CASE WHEN ActionId=64 THEN 1 ELSE 0 END) AS myWalletPayLoanWithCreditCard,
	   SUM(CASE WHEN ActionId=65 THEN 1 ELSE 0 END) AS fieldPasted,
	   SUM(CASE WHEN ActionId=66 THEN 1 ELSE 0 END) AS budgetToolOpened,
	   SUM(CASE WHEN ActionId=67 THEN 1 ELSE 0 END) AS budgetToolViewMyMoney,
	   SUM(CASE WHEN ActionId=68 THEN 1 ELSE 0 END) AS budgetToolViewMyAccounts,
	   SUM(CASE WHEN ActionId=69 THEN 1 ELSE 0 END) AS budgetToolViewMyTransactions,
	   SUM(CASE WHEN ActionId=70 THEN 1 ELSE 0 END) AS creditReportOpened,
	   SUM(CASE WHEN ActionId=71 THEN 1 ELSE 0 END) AS creditReportViewPersonalDetails,
	   SUM(CASE WHEN ActionId=72 THEN 1 ELSE 0 END) AS creditReportViewCreditOverview,
	   SUM(CASE WHEN ActionId=73 THEN 1 ELSE 0 END) AS creditReportViewMyAccounts,
	   SUM(CASE WHEN ActionId=74 THEN 1 ELSE 0 END) AS creditReportViewCPAAccounts,
	   SUM(CASE WHEN ActionId=75 THEN 1 ELSE 0 END) AS creditReportViewNLRAccount,
	   SUM(CASE WHEN ActionId=76 THEN 1 ELSE 0 END) AS creditReportViewPreviousEnquiries,
	   SUM(CASE WHEN ActionId=77 THEN 1 ELSE 0 END) AS creditReportViewJudgements,
	   SUM(CASE WHEN ActionId=83 THEN 1 ELSE 0 END) AS leadRegistered,
	   SUM(CASE WHEN ActionId=84 THEN 1 ELSE 0 END) AS leadLanding,
	   SUM(CASE WHEN ActionId=85 THEN 1 ELSE 0 END) AS viewLoanContract,
	   SUM(CASE WHEN ActionId=87 THEN 1 ELSE 0 END) AS selfHelpNotice,
	   SUM(CASE WHEN ActionId=88 THEN 1 ELSE 0 END) AS selfHelpQuestionList,
	   SUM(CASE WHEN ActionId=89 THEN 1 ELSE 0 END) AS selfHelpSuccess,
	   SUM(CASE WHEN ActionId=90 THEN 1 ELSE 0 END) AS selfHelpPaymentDate,
	   SUM(CASE WHEN ActionId=91 THEN 1 ELSE 0 END) AS selfHelpBankingDetails,
	   SUM(CASE WHEN ActionId=92 THEN 1 ELSE 0 END) AS selfHelpSimpleQuestion,
	   SUM(CASE WHEN ActionId=93 THEN 1 ELSE 0 END) AS appOtpSent,
	   SUM(CASE WHEN ActionId=94 THEN 1 ELSE 0 END) AS appOtpResend,
	   SUM(CASE WHEN ActionId=95 THEN 1 ELSE 0 END) AS appOtpAccepted,
	   SUM(CASE WHEN ActionId=96 THEN 1 ELSE 0 END) AS appOtpRejected
        FROM Atlas_Analytics..ClientActions ca 
        where ca.ActionDate >='{0}' and ca.ActionDate <'{1}'
        Group by SessionId'''.format(from_date,to_date)
        
df = pd.read_sql_query(the_SQL, cnxn, index_col = 'SessionId')
#%%

#%%
# determine optimal amount of clusters
#set seed for repeatable results
#seed 11 k=15 clusters silhouette score 0.24089004302045561
#seed 1 k=2 clusters silh sc 0.195950227414
#seed 1234 k=20 0.199782989696
seed = 0
# set range of amount of clusters
K = range(2,19)
km = [KMeans(init='k-means++', n_clusters=k, n_init = 100, random_state=seed).fit(df) for k in K]
centroids = [k.cluster_centers_ for k in km]
D_k = [cdist(df, cent, 'euclidean') for cent in centroids]
cIdx = [np.argmin(D,axis=1) for D in D_k]
dist = [np.min(D,axis=1) for D in D_k]
# metric one
avgWithinSS = [sum(d)/df.shape[0] for d in dist]
#%%
# Total with-in sum of square - two
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(df)**2)/df.shape[0]
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
    sillybilly.append(silhouette_score(df.as_matrix(), k.labels_, metric='euclidean', sample_size=10000, n_jobs=5))
    print(sillybilly[i])
#    print(k.labels_)
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
pca1.fit(df)

plt.figure(1, figsize=(16, 12))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca1.explained_variance_, linewidth=2)
plt.axis('tight')
plt.grid(True)
plt.xlabel('n_components')
plt.ylabel('explained_variance_')

#%%
# Choose components above and then do the following
sillybilly=[]
K = [2,3,4,5,6,7,8,9,10,11,12]
for n_comp in K:
    pca = PCA(n_components=n_comp).fit(df)
    kmeans_clust = KMeans(init=pca.components_, n_clusters=n_comp, n_init=1).fit(df)
    sillybilly.append(silhouette_score(df.as_matrix(), k.labels_, metric='euclidean', sample_size=10000, n_jobs=5))
    print(sillybilly)

#%% best was 3 clusters, 5 also looks good
seed = 0
kmeans_clust = KMeans(init='k-means++', n_clusters=5, n_init=100,random_state=seed,verbose=3)

kmeans_clust.fit(df)
#%%
# Write clusters back to dataset
df_clust = df.copy()
df_clust['cluster'] = kmeans_clust.labels_
df_clust.cluster.value_counts()