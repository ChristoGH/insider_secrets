# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 12:33:45 2016

@author: Christiaan
"""
#%%
import pandas as pd
import numpy as np
import geoip2.database
import pyodbc
#%%
cnxn = pyodbc.connect(driver='{SQL Server}', server='197.96.22.2,5595',\
                              database='za_FinCloud', UID='AiUser', PWD='AiUser.Psibyl11', autocommit=True)

SQL = '''
        SELECT c.RefId, c.IdNumber,
               b.[UserOriginIp],
               d.[AddressLine4],
               d.[PostalCode],
               d.region
        FROM dbo.Clients c
             INNER JOIN
        (
            SELECT clientrefid,
                   COUNT(refid) AS lncnt
            FROM dbo.Loans
            WHERE LoanEffective < '2016-07-01'
                  AND productrefid IN(1, 2, 3, 13, 14, 15, 16, 17, 19, 20, 21, 24)
            GROUP BY clientrefid
        ) a ON a.ClientRefId = c.RefId
             LEFT JOIN
        (
            SELECT *
            FROM
            (
                SELECT *,
                       RANK() OVER(PARTITION BY clientrefid ORDER BY createdate DESC) AS ranknum
                FROM dbo.LoanApplications
            ) piet
            WHERE ranknum = 1
        ) b ON b.ClientRefId = c.RefId
             LEFT JOIN
        (
            SELECT a.*,
                   COALESCE(p.Provincename, '0') AS region,
                   RANK() OVER(PARTITION BY owningrefid ORDER BY a.refid deSC) AS ranknum
            FROM dbo.Address a
                 LEFT JOIN provinces p ON p.refid = a.provincelookupkey
        	    where a.addresstype = 0
        ) d ON d.OwningRefId = c.RefId
               AND (d.Addressline4 <> 'N/A'
                    OR d.PostalCode <> '0000'
                    OR d.region <> '0')
        WHERE d.ranknum = 1
        ORDER BY c.RefId;'''

df_nodup = pd.read_sql_query(SQL, cnxn, index_col = 'RefId')
cnxn.close()
#%%
# duplicates
#len(df.RefId.unique())
#df[df.RefId.duplicated(keep=False)]
#idDropper = [56636,55809,48502,46978,43988,43242,42416,42076]
#df_nodup = df.drop(df.index[idDropper])
#len(df_nodup.RefId.unique())
#df_nodup[df_nodup.RefId.duplicated(keep=False)]
#df_nodup = df_nodup.drop(df_nodup.index[idDropper])
#%%
# kry eers mense wat wel provinsie ingevul het (addressline 4)
prnams = pd.read_excel('data/provinceNames.xlsx')
prnams = prnams.set_index(['prind'])
df_nodup['pr_nm'] = df_nodup.AddressLine4.str.lower().map(prnams.pr_nm)

#%% as provinsie inligting in region beskikbaar is, gebruik dit
df_nodup.loc[(df_nodup.pr_nm.isnull()) & (df_nodup.region != '0'),'pr_nm']=df_nodup.loc[(df_nodup.pr_nm.isnull()) & (df_nodup.region != '0'),'region']
#%%
# map postal code with function
def pctopr(x):        
    y=None    
    if x == 0:                      y = None    
    elif x > 0 and x <300:          y = 'Gauteng'
    elif x < 500:                   y = 'North west'
    elif x < 1000:                  y = 'Limpopo'
    elif x < 1400:                  y = 'Mpumalanga'
    elif x < 2200:                  y = 'Gauteng'
    elif x < 2500:                  y = 'Mpumalanga'
    elif x < 2900:                  y = 'North West'
    elif x < 4731:                  y = 'KwaZulu-Natal'
    elif x < 6500:                  y = 'Eastern Cape'
    elif x < 8100:                  y = 'Western Cape'
    elif x < 9000:                  y = 'Northern Cape'
    elif x >= 9300 and x < 10000:   y = 'Free State'
    return y
#%%
df_nodup.pr_nm.fillna(pd.to_numeric(df_nodup.PostalCode,errors='coerce').map(pctopr),inplace=True)

#%%
#map IP
reader = geoip2.database.Reader('data/GeoLite2-City.mmdb')

ipmap = []
for x in [x for x in df_nodup.UserOriginIp.unique()]:
    y = 'niks'    
    try:
        response = reader.city(x)
        y = response.subdivisions.most_specific.name
    except: 
        y= 'unknown'
        pass
    ipmap.append([x,y])  
    
ipmap1 = pd.DataFrame(ipmap)

ipmap1.columns = ['ip','region']

sum(ipmap1.region.notnull())

sum(ipmap1.region == 'unknown')

ipmap2 = ipmap1.set_index(['ip'])

df_nodup.pr_nm.fillna(df_nodup.UserOriginIp.map(ipmap2.region),inplace=True)


#%%

df_nodup.loc[df_nodup.pr_nm == 'unknown','pr_nm']=None
#%% verwyder soortgelyke provinsiename
df_nodup.pr_nm.unique()

df_nodup.loc[df_nodup.pr_nm=='Province of the Western Cape',['pr_nm']] = 'Western Cape'
df_nodup.loc[df_nodup.pr_nm=='Orange Free State',['pr_nm']] = 'Free State'
df_nodup.loc[df_nodup.pr_nm=='North west',['pr_nm']] = 'North West'
df_nodup.loc[df_nodup.pr_nm=='The Western Cape',['pr_nm']] = 'Western Cape'
df_nodup.loc[df_nodup.pr_nm=='The Eastern Cape',['pr_nm']] = 'Eastern Cape'
df_nodup.loc[df_nodup.pr_nm=='Province of North West',['pr_nm']] = 'North West'
df_nodup.loc[df_nodup.pr_nm=='The Free State',['pr_nm']] = 'Free State'
#%%
# Remove province names that result from foreign IPs
df_nodup.pr_nm.unique()
  
wrongip = ['North Carolina', 'Île-de-France', 'Georgia', 'Paris',
       'California','England', 'Schleswig-Holstein',
       'Dublin City', 'Bavaria', 'Hesse', 'Illinois', 'Bangkok',
       'Plaines Wilhems District', 'Tamil Nadu', 'Texas', 'New Jersey',
       'Bracknell Forest', 'Doncaster', 'Zurich', 'Skåne', 'New York',
       'Massachusetts', 'Indiana', "Mykolayivs'ka Oblast'", 'Birmingham',
       'Gloucestershire', 'Western Australia', 'Cairo Governorate',
       'Khomas', 'Västra Götaland', 'Ar Riyāḑ', 'South East District',
       'Muhafazat al Gharbiyah', 'Down', 'North Holland', 'Aberdeen City',
       'Nairobi Province', 'Maryland', 'Devon', 'Seine-et-Marne',
       'Minnesota', 'Michigan', 'Missouri', 'Cidade de Maputo',
       'New South Wales', 'Virginia', 'Brighton and Hove', 'Stockholm',
       'Bournemouth', 'Reading', 'Hamburg', 'Baden-Württemberg Region',
       'Arizona', 'Dubai', 'Rhône', 'Suffolk', 'Brussels Capital',
       'Capital Region', 'Seine-Saint-Denis', 'Hauts-de-Seine',
       'Judetul Salaj', 'Pennsylvania', 'Turin', 'Hertfordshire', 'Wirral',
       'Val-de-Marne','Cidade de Maputo']
       
wrongip = pd.DataFrame(list(set(wrongip)))
wrongip['mapto']='unknown'
wrongip.columns = ['wrip','mapto']
wrongip = wrongip.set_index(['wrip'])
df_nodup['pr_nm2']=df_nodup.pr_nm.map(wrongip.mapto)
df_nodup.pr_nm2.fillna(df_nodup.pr_nm, inplace=True)

#%%

# map city
# Major Caveat : City names duplicated across provinces -
# So just fill the last few nulls and hope for the best
cr = pd.read_excel('data/SouthAfricanCities.xlsx')
cr1 = cr.drop_duplicates(['City'])
crmap1 = cr1[['City','ProvinceName']]
crmap1 = crmap1.set_index(['City'])
df_nodup['AddressLine4']=df_nodup.AddressLine4.str.lower()
df_nodup.pr_nm.fillna(df_nodup.AddressLine4.map(crmap1.ProvinceName),inplace=True)

#%%
# Replace remaining NAs with unknown
df_nodup.pr_nm.fillna('unknown', inplace=True)
#%%
df = df_nodup[['IdNumber','pr_nm2']].copy()

#%%
df.to_excel('data/za_clients_with_region.xlsx')