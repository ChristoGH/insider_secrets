#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:26:06 2019

@author: lnr-ai
"""

import os
os.chdir('/home/lnr-ai/krisjan/insider_secrets/')
#%%
import pandas as pd
#%%
df = pd.read_hdf('eateries/client_lens_cb_pbcvm.h5',key='client_data')
df_trans = pd.read_csv('eateries/eateries_transactions.csv')
df_trans.drop(['Unnamed: 0'],axis=1,inplace=True)
df_trans_agg = pd.DataFrame(df_trans.groupby(['Dedupegroup','companyname']).TransactionAmount.sum())
df_trans_sum = df_trans_agg.unstack(-1)
df_trans_sum.columns = df_trans_sum.columns.droplevel()
df_trans_agg = pd.DataFrame(df_trans.groupby(['Dedupegroup','companyname']).TransactionAmount.count())
df_trans_cnt = df_trans_agg.unstack(-1)
df_trans_cnt.columns = df_trans_cnt.columns.droplevel()

df_cluster = df.merge(df_trans_sum,left_index=True,right_index=True)
df_cluster = df_cluster.merge(df_trans_cnt,left_index=True,right_index=True,suffixes=['_amt','_cnt'])
#%%
df_cluster['client_age'] = (pd.Timestamp.now().year + pd.Timestamp.now().month/12) -\
                            (df_cluster.BirthDate.dt.year + df_cluster.BirthDate.dt.month/12)
del df_cluster['BirthDate']
df_cluster.to_hdf('eateries/cluster_data',key='df_raw',complevel=9)
#%%
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.externals import joblib
#%%
le_race = LabelEncoder()
df_cluster.Race = le_race.fit_transform(df_cluster.Race)
joblib.dump(le_race, 'eateries/race_enc.joblib')

le_lang = LabelEncoder()
df_cluster.Language = le_lang.fit_transform(df_cluster.Language)
joblib.dump(le_lang, 'eateries/language_enc.joblib')

le_gender = LabelEncoder()
df_cluster.Gender = le_gender.fit_transform(df_cluster.Gender)
joblib.dump(le_gender, 'eateries/gender_enc.joblib')

le_marital = LabelEncoder()
df_cluster.MaritalStatus = le_marital.fit_transform(df_cluster.MaritalStatus)
joblib.dump(le_marital, 'eateries/marital_enc.joblib')

le_province = LabelEncoder()
df_cluster.Province.fillna('NA',inplace=True)
df_cluster.Province = le_province.fit_transform(df_cluster.Province)
joblib.dump(le_province, 'eateries/province_enc.joblib')

del df_cluster['Seg_l2_str']
le_seg_l3 = LabelEncoder()
df_cluster.Seg_l3_str = le_seg_l3.fit_transform(df_cluster.Seg_l3_str)
joblib.dump(le_seg_l3, 'eateries/seg_3_enc.joblib')

df_cluster.fillna(0,inplace=True)

ss = StandardScaler()
df_analysis = pd.DataFrame(ss.fit_transform(df_cluster),index=df_cluster.index,columns=df_cluster.columns)
joblib.dump(ss, 'eateries/standard_scaler.joblib')
#%%
df_cluster.to_hdf('eateries/cluster_data',key='df_numeric',complevel=9)
df_analysis.to_hdf('eateries/cluster_data',key='df_scaled',complevel=9)