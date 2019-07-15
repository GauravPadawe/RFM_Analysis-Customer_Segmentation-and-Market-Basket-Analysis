#!/usr/bin/env python
# coding: utf-8

# # Market Basket Analysis - Apriori Algorithm
# 
# ### Domain : Retail
# 
# ### Table of Contents
# 
# #### 1. **Information**
#     - Questions Arising
#     - Objective
# 
# #### 2. **Loading Dataset**
#     - Importing packages
#     - Reading Data
#     - Shape of data
#     - Dtype
# 
# #### 3. **Data Cleaning & EDA**
#     - Checking Null values
#     - EDA
# 
# #### 4. **Apriori Algorithm**
#     - Pivot table (Forming a basket)
#     - Apriori Implementation
#     - Association Rules
# 
# #### 5. **Conclusion**
# 
# 
# ### Questions Arising :
# 
# - Based on Monetary Value or spendings , can we find Top 10 Customers ?
# 
# 
# - What are the most expensive products ? 
# 
# ### Objective :
# 
# - Perform Market Basket Analysis on UK Dataset for which we performed RFM Analysis in First Part.
# 
# 
# - Mining frequent Item sets to generate association rules.

# ### Importing Required Packages

# In[1]:


#importing required packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().system('pip install mlxtend')


# - Above we can see that we are installing new package called "mlxtend (machine learning extensions)."
# 
# 
# - **Documentation : http://rasbt.github.io/mlxtend/**

# In[2]:


#reading data

df = pd.read_csv('uk_retail_data.csv', header=0, parse_dates=True)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']) #conversion to datetime format
df.drop(df.columns[0], axis=1, inplace=True)
df.head()


# In[3]:


#shape of data

print ('No. of Records :',df.shape[0])
print ('No. of Features :', df.shape[1])


# ### Data Cleansing & EDA

# In[4]:


#dtypes

df.info()


# In[5]:


#Missing values

df.isnull().sum()


# In[6]:


#Max and Min Quantities

print ('Minimum Quantity :', min(df['Quantity']))
print ('Maximum Quantities :', max(df['Quantity']))


# **Observations :**
# 
# - Minimum Quantities purchased by a customer is 1 Unit.
# 
# 
# - Maximum Quantities purchased are 80995 Units.

# In[7]:


#Unique Transactions, products, customers

print ('Unique No. of Transactions :', len(pd.unique(df['InvoiceNo'])))
print ('Unique No. of Products :', len(pd.unique(df['Description'])))
print ('Unique No. of Customers :', len(pd.unique(df['CustomerID'])))


# **Observations :**
# 
# - We've total 16649 Unique Invoices / Transactions. It could be by repeating customer or one-time customer.
# 
# 
# - Unique Items found in our dataset are 3844.
# 
# 
# - Unique Identified Customers count is 3921.

# In[8]:


#best customers

sns.set_context('talk')

best_cust = df.groupby('CustomerID')['Total_price'].agg({'Total_price':'sum'})
best_cust = best_cust.sort_values(by='Total_price', ascending=False)

best_cust.head(10).plot(kind='bar', legend=False, color='grey')
plt.ylabel('Total Price Spent')
plt.title('Top 10 Customers by Monetary Value')


# **Observations :**
# 
# - Above viz. depicts Monetary amount spent by Individual Customers.
# 
# 
# - These are our top 10 consumers based on amount of money they spent.
# 
# 
# - Here we are not considering Recency & Frequency , just Monetary amount.

# In[9]:


#Range of UnitPrice

sns.boxplot(df['UnitPrice'])
plt.title('Distribution of Unit Price')


# **Observations :**
# 
# - The price distribution ranges from 0 to 8000.
# 
# 
# - We can explore these values outliers.

# In[10]:


#Top 10 expensive Products

df.sort_values(by='UnitPrice', ascending=False)[:10][['Description', 'UnitPrice']]


# **Observations :**
# 
# - We've most expensive product as "POSTAGE", which must be a Postage charge paid by Consumer on some product.
# 
# 
# - While, Manual could be service charge for Installation / Assembling a product. Since, we don not have any information of Retail company and product we could just best guess based on information we've in-hand.
# 
# 
# - We can drop Manual, Postage and Dotcom Postage while creation of basket.

# In[11]:


#slicing out UnitPrice with 0
df = df[df.UnitPrice != 0]

#Top 10 cheap products
df.sort_values(by='UnitPrice', ascending=True)[:10][['Description', 'UnitPrice']]


# **Observations :**
# 
# - We had dropped columns with 0 Unit Price as they can be free or sub-products which could be a part of Major product. 
# 
# 
# - Above are some cheapest games.

# ### Apriori Algorithm
# 
# -  We'll form a basket to apply Apriori Algorithm.
# 
# 
# - We'll group on Invoice and use products as our columns & values as Quantity.
# 
# 
# - We'll drop Postage, Bank Charges, Dotcom Postage, Manual as they won't make any sense.
# 
# 
# - We had encode the Quantities that > 1 to 1.

# In[ ]:


#forming a basket

basket = pd.pivot_table(data=df, index='InvoiceNo', columns='Description', values='Quantity', fill_value=0)
#basket[10:20]


# In[13]:


#value encoder

def encoder(data):
    if data >= 1:
        return 1
    else:
        return 0

#dropping POSTAGE
basket_enc = basket.applymap(encoder)
basket_enc.drop(['POSTAGE', 'Manual', 'DOTCOM POSTAGE', 'Bank Charges'], axis=1, inplace=True)
basket_enc[10:20]


# **Working of Apriori Algorithm**
# 
# - Apriori is an algorithm for frequent item set mining and association rule learning over transactional databases.
# 
# 
# - It proceeds by identifying the frequent individual items in the database and extending them to larger and larger item sets as long as those item sets appear sufficiently often in the database.
# 
# 
# - Support, Confidence & Lift are our metrics to buid Association Rules.
# 
# 
# ![alt text](https://miro.medium.com/max/1067/1*--iUPe_DtzKdongjqZ2lOg.png)

# In[19]:


#implementing Apriori @ support 0.025

from mlxtend.frequent_patterns import apriori, association_rules

freq_itemset = apriori(df=basket_enc, min_support=0.025, use_colnames=True, )
freq_itemset.sort_values(by='support', ascending=False)


# - By keeping our Support threshold to 0.025 we get some better results for Association Rules.
# 
# 
# - We've got 149 records of Frequent Individual Items and Frequent Paired / Brought together items.
# 
# 
# - Let's implement association rules now.

# In[20]:


#calculating confidence & lift metrics

rules = association_rules(df=freq_itemset, metric='lift', min_threshold=1)
rules[['antecedents','consequents', 'support','confidence','lift']]


# - In practice it is considered that Lift value above 1 is best Association. 
# 
# 
# - For example, If a customer bought (ALARM CLOCK BAKELIKE RED ) then he will likely purchase (ALARM CLOCK BAKELIKE GREEN) as well.
# 
# 
# - For our conserdation we'll slice the values having Confidence >= 0.5 & Lift >= 5.

# In[22]:


rules[(rules.confidence >= 0.5) & (rules.lift >= 5)][['antecedents','consequents', 'support','confidence','lift']].sort_values(by='lift', ascending=False)


# - By above Slicing & Sorting we can say that above products are highly associated.
# 
# 
# - If a person buys product A (Antecedents) then he/she will like opt for product C (Consequents) because Confidence is above 0.50 & Lift above 5.
# 
# 
# - With this we conclude our Market Basket Analysis for UK Data.

# **Conclusion :**
# 
# - We had performed Market Basket Analysis and interpreted the results.
# 
# 
# - We can focus on products with confdence 0.5 and lift value greater than equals 1 but less than 5 because life value greater than 1 is condered to be acceptable.
