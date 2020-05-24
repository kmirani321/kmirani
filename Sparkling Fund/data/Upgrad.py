#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


companies = pd.read_csv("./data/companies.txt", sep="\t", encoding = "unicode_escape")
rounds2 = pd.read_csv("./data/rounds2.csv",encoding='unicode_escape')
mapping = pd.read_csv("./data/mapping.csv", encoding = "unicode_escape")


# In[3]:


rounds2["company_permalink"] = rounds2["company_permalink"].str.lower()
companies["permalink"] = companies["permalink"].str.lower() # formatting to identify duplicacy and uniqueness


# In[4]:


unirounds2=len(rounds2["company_permalink"].unique())
unirounds2 # no of unique companies in rounds2


# In[5]:


unicompanies=len(companies.permalink.unique())
unicompanies # of unique companies in companies.csv


# In[6]:


master_frame=pd.merge(rounds2,companies,how="left",left_on="company_permalink",right_on="permalink") #master data frame merge


# In[12]:


master_frame.isnull().sum() # sum of mssing values in rows (percentage)
round(100*(master_frame.isnull().sum()/len(master_frame.index)), 2)


# In[ ]:


master_frame = master_frame.drop('funding_round_code', axis=1)
master_frame = master_frame.drop('founded_at', axis=1)
master_frame = master_frame.drop('homepage_url', axis=1) # deleting non required columns


# In[15]:


master_frame=master_frame[~master_frame.raised_amount_usd.isnull()] # cleaning the null in raised amount as this is important field
master_frame['raised_amount_usd'].isnull().sum()


# In[16]:


master_frame=master_frame[~master_frame.country_code.isnull()]
master_frame=master_frame[~master_frame.category_list.isnull()] # cleaning other useful important columns 


# In[17]:


round(100*(master_frame.isnull().sum()/len(master_frame.index)), 2)


# In[18]:


master_frame = master_frame.drop('state_code', axis=1)
master_frame = master_frame.drop('region', axis=1)
master_frame = master_frame.drop('city', axis=1) # removing other non essential columns


# In[19]:


round(100*(master_frame.isnull().sum()/len(master_frame.index)), 2)


# In[20]:


len(master_frame.index)/114942  # we have lost 33% of data till now :)


# In[21]:


master_frame=master_frame[master_frame['funding_round_type'].isin(['seed','angel','venture','private_equity'])] # filtering the type of investment preferred by SF 


# In[25]:


ventures = master_frame[master_frame["funding_round_type"]=="venture"]
countrytoinvest = ventures.groupby('country_code')
countrytoinvest['raised_amount_usd'].sum().sort_values(ascending=False).head(10)


# In[29]:


top9 = ventures[ventures['country_code'].isin(["USA","CHN","GBR","IND","CAN","FRA","ISR","DEU","JPN"])]


# In[28]:


top9=top9[~top9.category_list.isnull()]


# In[30]:


top9.head()


# In[39]:


top9['primary_sector'] =top9['category_list'].astype(str).apply(lambda x: x.split('|')[0])


# In[40]:



top9.shape


# In[41]:


mapping=mapping[~mapping.category_list.isnull()]
mapping.head()


# In[42]:


def correctMapping0(category):
## Function to correct the mapping data.It can be observed fromt the data 
##that at many places in the category_class column the 'na' is misprinded as '0'
## Eg. 'Analysis' is present as 'A0lysis'
    if '0' in category:
        if category.startswith("0"):    # Handle 0notechnology case with capital Na
            return category[:category.find('0')]+'Na'+category[category.find('0')+1:]
        elif category.endswith("0"):    # ignores the 2.0 case with 0 at the end of the string
            return category
        else:
            return category[:category.find('0')]+'na'+category[category.find('0')+1:]
    else:
        return category


# In[43]:


mapping['category_list']=list(map(correctMapping0,mapping['category_list']))


# In[44]:



m=pd.melt(mapping, id_vars=['category_list'], var_name=['main_sector'])
m=m[m.value==1]
m=m.drop('value',axis=1)
m.shape


# In[46]:


top9=pd.merge(top9,m,how="left",left_on="primary_sector",right_on="category_list")
top9=top9.drop('category_list_y',axis=1)


# In[48]:


top9.head()


# In[51]:


top9=top9[~(top9['main_sector_y'].isnull())]


# In[52]:


top9.head()


# In[55]:


D1=top9[top9['country_code']=='USA'] ## USA
D1 = D1[(D1['raised_amount_usd'] >= 5000000) & (D1['raised_amount_usd'] <= 15000000)]
D1.raised_amount_usd.sum()


# In[59]:


D1_by_sector=D1.groupby('main_sector_y')
D1_by_sector['raised_amount_usd'].sum().sort_values(ascending = False).head()


# In[61]:


D1[D1['main_sector_y']=='Others'].groupby(['main_sector_y','permalink']).raised_amount_usd.sum().sort_values(ascending=False).head()


# In[62]:


# 2nd best sector to invest
D1[D1['main_sector_y']=='Social, Finance, Analytics, Advertising'].groupby(['main_sector_y','permalink']).raised_amount_usd.sum().sort_values(ascending=False).head()


# In[66]:


D2=top9[top9['country_code']=='CHN']  ## china
D2 = D2[(D2['raised_amount_usd'] >= 5000000) & (D2['raised_amount_usd'] <= 15000000)]
D2.raised_amount_usd.sum()


# In[68]:


D2_by_sector=D2.groupby('main_sector_y')
D2_by_sector['raised_amount_usd'].sum().sort_values(ascending = False)


# In[70]:


D2[D2['main_sector_y']=='Others'].groupby(['main_sector_y','permalink']).raised_amount_usd.sum().sort_values(ascending=False).head()


# In[73]:


D2[D2['main_sector_y']=='Social, Finance, Analytics, Advertising'].groupby(['main_sector_y','permalink']).raised_amount_usd.sum().sort_values(ascending=False).head()


# In[74]:


D3=top9[top9['country_code']=='GBR']
D3 = D3[(D3['raised_amount_usd'] >= 5000000) & (D3['raised_amount_usd'] <= 15000000)]
D3.raised_amount_usd.sum()


# In[75]:


D3_by_sector=D3.groupby('main_sector_y')
D3_by_sector['raised_amount_usd'].sum().sort_values(ascending = False)


# In[76]:



D3[D3['main_sector_y']=='Social, Finance, Analytics, Advertising'].groupby(['main_sector_y','permalink']).raised_amount_usd.sum().sort_values(ascending=False).head()


# In[77]:


# bar plot
plt.figure(figsize=(10,5))
g=sns.barplot(x='funding_round_type', y='raised_amount_usd', data=master_frame)
#g.set_ylim(0, 20000000)
#g.set_yscale('log')
g.set(xlabel='Funding Type', ylabel='Raised Amount   ( 1 Unit = 10M USD)')
g.set_title('Funding Type Analysis',fontsize =18)

plt.axhline(5000000, color='green')
plt.axhline(15000000, color='red')


plt.show()


# In[78]:


plt.figure(figsize=(10,5))
c=sns.barplot(x='country_code', y='raised_amount_usd', data=top9, estimator=np.sum,color=(0.2, 0.4, 0.7, 0.6))
#c.set_ylim(0, 100000000000)
c.set_yscale('log')
c.set(xlabel='Funding Type', ylabel='Raised Amount')
c.set_title('Country Analysis',fontsize =18)
plt.show()


# In[83]:


master_D=D1[D1['main_sector_y'].isin(['Others','Social, Finance, Analytics, Advertising','Cleantech / Semiconductors'])]
master_D=master_D.append(D2[D2['main_sector_y'].isin(['Others','Social, Finance, Analytics, Advertising','News, Search and Messaging'])], ignore_index=True)
master_D=master_D.append(D3[D3['main_sector_y'].isin(['Others','Social, Finance, Analytics, Advertising','News, Search and Messaging'])], ignore_index=True)


# In[85]:



# set figure size for larger figure
plt.figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')

# specify hue="categorical_variable"
s=sns.barplot(x='country_code', y='raised_amount_usd', hue="main_sector_y", data=master_D,estimator=lambda x: len(x))
s.set(xlabel='Country', ylabel='Number of Investments')
s.set_title('Investments in top3 sectors of top 3 countries',fontsize =18)
plt.show()


# In[ ]:




