#!/usr/bin/env python
# coding: utf-8

# # <center><b><font color='olive'>RFM Analysis</b></center>
# 
# ## <center><b><font color='magenta'>The Key to Understanding Customer Buying Behavior</b></center>

# ## RFM Customer segmentation
# 
#    - **RECENCY (R)**: Time since last purchase.</b>
#    - **FREQUENCY (F)**: Total number of purchases.
#    - **MONETARY VALUE (M)**: Total monetary value.<b>
# 
# The goal of RFM Analysis is to segment/group customers based on their buying behavior. We need to understand the historical behavior of individual customers for each RFM factor. We then rank customers based on each individual RFM factor, and finally pull all the factors together to create RFM segments for targeted marketing. 
# 
# - Can you identify your best customers?
# - Do you know who your worst customers are?
# - Do you know which customers you just lost, and which ones you’re about to lose?
# - Can you identify loyal customers who buy often, but spend very little?
# - Can you target customers who are willing to spend the most at your store?

# ### Importing required Modules

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_excel('retail.xlsx', header=0, parse_dates=True)
df.head()


# ### EDA

# In[3]:


df.info()


# - The above results shows us that features with respective data types : 
#     - (1) x datetime64
#     - (2) x float64
#     - (1) x int64
#     - (4) x object

# In[4]:


df.isnull().sum()


# - The above null value analysis gives us sum of total null values in given feature.
#     - We can see that ***Description*** has 1454 NaN values.
#     - We can see that ***CustomerID*** has 135080 NaN values.

# In[5]:


df[['Quantity', 'UnitPrice']].describe()


# - There are negative values in ***Quantity*** and ***UnitPrice*** which is quite bizzare. 

# In[6]:


plt.figure(figsize=(15,7))
sns.set_context('talk')
sns.countplot(df.Country, palette='seismic')
plt.title('Countries Involved')
plt.xticks(rotation=90)


# - We got data of 38 countries which includes **North/South American, European, Middle East and other** countries.
# - Also, 90% of data is from **UK** so we'll conduct our RFM analysis on UK. Also, other reason is we cannot perform RFM analysis on multiple regions because as country changes, the behavior of customers , currency, lifestyle, products, etc changes too.

# In[7]:


uk = df[df.Country == 'United Kingdom']
uk.head()


# In[8]:


uk.shape


# - We got 495478 examples of **UK**

# In[9]:


uk.isnull().sum()


# - Missing values in UK data is :
#     - Description : 1454
#     - CustomerID : 133600

# In[10]:


#dropping null rows

uk = uk[pd.notnull(uk['CustomerID'])]
uk.isnull().sum()


#   - **Why did we drop rows with null values ?**
#   - Ans.: Because we've missing values in **CustomerID**, we cannot replace missing values of ID columns because that is a false approach, it can bias a data towards particular group of customers , which can lead to false analysis. 

# In[11]:


#% negative values in UK dataset

print (round(len(uk[uk['Quantity'] < 0])/len(uk) * 100),'% Negative Values')


# - We got **2% of Negative values** in the UK dataset.

# ### Descriptive Statistics

# In[12]:


uk = uk[uk.Quantity > 0]
uk[['Quantity', 'UnitPrice']].describe()


# - Let us only consider values which are Positive and focus on further analysis.
# - Above descriptive statistics tells us that Min. & Max. Quantity is 1 & 80995 while UnitPrice has Min. & Max. 0 & 8142.75.
# - We can observe that some customer has checked out with 80995 units of particular/single product but it is not necessary that it is more frequent customer because, let us assume that there was a stock clearance sale ran by store and the customer got to know it from advertises or marketing, which drove his attention towards our store and he opted out for the required product in bulk and as a result we could get such huge variation in data.

# In[13]:


uk.shape


# - In UK data (after cleaning) , we got 354345 examples for our RFM analysis.

# In[ ]:


uk['Total_price'] = uk.Quantity * uk.UnitPrice


# - We can do feature enginnering by taking product of No. of Quantities and Unit Price which will further helpful for Monetary Analysis.

# In[15]:


#min and max invoice date

df['InvoiceDate'].min(), df['InvoiceDate'].max()


# - We have data ranging from **12/2010** to **12/2011** which is 1 year of information we got under roof.

# In[16]:


import datetime as dt

next_day = df['InvoiceDate'].max() + dt.timedelta(1)
next_day


# - We need to add a day to last date in out data.
# - To understand this we need to revisit the defination of **Recency (R) , i.e, Time since last purchase**. We calculate Recency by subtracting current date with recorded date. The lesser the difference in days (lesser Gap), the recent was the customer found checking-in to our store. The bigger the difference, the larger is the gap since he last visited.
# - Now, if we consider to subtract today's date with the customers who visited today, that will give us 0. Neutral value won't add any value to our Recency (R) analysis so we must add a day to current date.

# ### No. of Sales per day and per hour in a day

# In[17]:


weekday = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Sunday']
plt.subplots_adjust(wspace=0.2, hspace=1)
plt.figure(figsize=(20,8))

plt.subplot(1,2,1)
sns.countplot(sorted(uk['InvoiceDate'].dt.weekday_name, key=weekday.index), palette='rainbow')
plt.xticks(rotation=90)
plt.title('Sale Count by Week day')

plt.subplot(1,2,2)
sns.countplot(uk['InvoiceDate'].dt.hour, palette='icefire')
plt.xticks(rotation=90)
plt.xlabel('Hour')
plt.title('Sale Count per Hour')


# - From above plot we can observe that most sale happens mostly on **Thursday and Wednesday**.
# - While in single day, most sales occur tend to occur between **11.00 to 15.00 hours.**

# ### Percentage of Repeated VS One-Time Consumers

# In[18]:


plt.figure(figsize=(7,7))

repeat_vs_one_time = np.where(uk.CustomerID.value_counts()>1, 'repeat', 'one_time')
repeat_vs_one_time = pd.DataFrame(repeat_vs_one_time)
repeat_vs_one_time = repeat_vs_one_time[repeat_vs_one_time.columns[0]].value_counts()
repeat_vs_one_time.plot(kind='pie', autopct='%1.1f%%')


# - The Percentage count of repeated customer is 98.2 while one-time customers are 1.8%

# ### Most & Least purchased products

# In[19]:


products = uk.groupby(['Description'])['Quantity'].sum().sort_values(ascending=False)
products.head(10)


# - We can see that **PAPER CRAFT , LITTLE BIRDIE, MEDIUM CERAMIC TOP STORAGE JAR, WORLD WAR 2 GLIDERS ASSTD DESIGNS, etc** are most purchased. 
# 
# - **Note** that most purchased product is not the most common product , remember we talked above about **Stock Clearance Sale** where a consumer purchases a sale/discounted product in bulk. These bulk purchase can fall into our **"Most Purchased Product"** category.

# In[20]:


#least purchased products
products.tail(10)


# - Above are some least purchased products

# ### RFM

# In[21]:


#Let us look at data again

uk.head()


# In[ ]:


# Converting DateTime to proper format

uk['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])


# - Let's create RFM Table now , for that we'll group our data on **CustomerID** and aggregrate the data such that
#     - Subtracting today's date with customer's recorded date will give us **Recency** of a Customer.
#     - Total number of purchases will give us **Frequency**
#     - And Summing up all the amount spent will give us **Monetary** 

# In[ ]:


rfm_table = uk.groupby('CustomerID').agg({'InvoiceDate' : lambda x: (next_day - x.max()).days,
                                          'InvoiceNo' : lambda x: len(x),
                                          'Total_price' : lambda x: x.sum()
                                         })


# In[24]:


rfm_table.info()


# - By grouping our data on **CustomerID** we an see that we get **3921 unique customers** who visited our retail unit over the period of time.

# In[ ]:


# Changing our column names to R, F & M.

rfm_table.rename(columns={'InvoiceDate' : 'Recency',
                          'InvoiceNo' : 'Frequency',
                          'Total_price' : 'Monetary'}, inplace=True)


# In[26]:


rfm_table.head()


# **For CustomerID 12346.0 :**
# 
#    - It's evident that Customer has visted our retail unit 326 days ago ,i.e, almost a **year** ago.
#    - We can note that the Customer has generated invoice only **once in those 326 days.**
#    - But, Monetary value is **77183.60**
#    
# 
# **For CustomerID 12347.0 :**
# 
#    - It's evident that this particular Customer has visted our retail unit couple of days ago.
#    - We can also note that the Customer has generated invoice **103 times**.
#    - And, Monetary value is **4196.01**

# In[27]:


#Let us look at what customer 12346 purchased

uk[uk['CustomerID'] == rfm_table.index[0]]


# - We can see that he puchased single product in bulk units maybe because it was a clearance sale.

# In[28]:


quartile = rfm_table.quantile(q=[0.25, 0.5, 0.75])
quartile


# - Above we have added the table in Quartile range , i.e, 25% - 50% - 75%
# - Below is the image to best describe Quartile Range.

# ![Image](https://www.lemnatec.com/wp-content/uploads/bell_curve_nir.jpg)

# In[29]:


quartile = quartile.to_dict()
quartile


# In[30]:


quartile['Recency'][0.25]


# Lowest recency, highest frequency and monetary are our best customers :
# - For recency a good customer would be a part of the lowest quartile designated as '1'. Lower recency states how recent customer shopped something in store.
# - For frequency and monetary a good customer would be a part of the highest quartile here designated as '1'. This states that higher frequency/monetary.
# - In short, R is inversely proportional to F & M.

# In[ ]:


#User defined functions

def Rscore(data, column, quartiles):
    if data <= quartiles[column][0.25]:
        return 1
    if data <= quartiles[column][0.5]:
        return 2
    if data <= quartiles[column][0.75]:
        return 3
    else:
        return 4
    
def FMscore(data, column, quartiles):
    if data <= quartiles[column][0.25]:
        return 4
    if data <= quartiles[column][0.5]:
        return 3
    if data <= quartiles[column][0.75]:
        return 2
    else:
        return 1


# - We had created user-defined functions here to assign respective scores based on which quartile data belongs to.

# ### Assigning RFM Scores

# In[ ]:


#Taking copy of RFM table

rfm_table_copy = rfm_table.copy()


# In[ ]:


#Applying Defined functions on specified columns
rfm_table_copy['r_quartile'] = rfm_table_copy['Recency'].apply(Rscore, args=('Recency', quartile))
rfm_table_copy['f_quartile'] = rfm_table_copy['Frequency'].apply(FMscore, args=('Frequency', quartile))
rfm_table_copy['m_quartile'] = rfm_table_copy['Monetary'].apply(FMscore, args=('Monetary', quartile))

#Resetting Index
rfm_table_copy.reset_index(inplace=True)

#Forming a new column which includes combined rfm score
rfm_table_copy['RFMscore'] = rfm_table_copy['r_quartile'].astype('str') + rfm_table_copy['f_quartile'].astype('str') + rfm_table_copy['m_quartile'].astype('str')


# In[34]:


rfm_table_copy.head(10)


# - Above we can see that we had nicely assigned RMF scores to respective customers.
# - In total, we've 4 quartiles. **25% - 50% - 75%** and other quartile has all the values which are **outliers.**

# ### Distribution of Recency, Frequency and Monetary

# In[35]:


sns.set_context('paper')
plt.subplots_adjust(hspace=1, wspace=1)
plt.figure(figsize=(20,15))

plt.subplot(3,1,1)
sns.distplot(rfm_table_copy['Recency'], color='r')

plt.subplot(3,1,2)
sns.distplot(rfm_table_copy['Frequency'], color='g')

plt.subplot(3,1,3)
sns.distplot(rfm_table_copy['Monetary'], color='b')


# - Above Viz. gives us brief idea on how our RFM data data is distributed. 

# ### Correlation heatmap

# In[36]:


#correlation plot

sns.set_context('talk')
sns.heatmap(rfm_table_copy[['Recency', 'Frequency', 'Monetary']].corr(), annot=True)
plt.title('Pearson Correlation')


# - Heatmaps are beautiful ways of representing Correlations.
# - Frequency and monetary value are positively correlated with each other implying an increase in frequency implies increase in monetary value.
# - Frequency and Recency are negatively correlated with each other.

# ### Targeting different customer segments

# ### (1) Best Customers
# 
# - These are the customers that bought recently, buy often and spend a lot. It’s likely that they will continue to do so.
# - Since they already like you so much, consider marketing to them without price incentives to preserve your profit margin.
# - Be sure to tell these customers about new products you carry, how to connect on social networks, and any loyalty programs or social media incentives you run.

# In[37]:


rfm_table_copy[rfm_table_copy['RFMscore'] == '111'].sort_values('Monetary', ascending=False).head(10)


# In[ ]:


rfm_table_copy['cust_type'] = np.where(rfm_table_copy['RFMscore'] == '111', 'Best_Cust', 'NA')


# ### (2) Big Spenders
# 
# - Big spenders have spent a lot of money over their lifetime as your customer.
# - They trust you enough to invest a lot in your products.
# - Considering marketing your most expensive products and top of the line models to this group.

# In[39]:


rfm_table_copy[rfm_table_copy['m_quartile'] == 1].sort_values('Monetary', ascending=False).head(10)


# In[40]:


big_spenders = rfm_table_copy[(rfm_table_copy['cust_type'] == 'NA') & (rfm_table_copy['m_quartile'] == 1)] #['cust_type'].replace('NA', 'Big_spenders')
rfm_table_copy['cust_type'][big_spenders.index] = 'Big_Spenders'


# ### (3) Loyal Ones
# 
# - Anyone with a high frequency should be considered loyal. This doesn’t mean they have necessarily bought recently, or that they spent a lot, though you could define that with your R and M factors.

# In[41]:


rfm_table_copy[rfm_table_copy['f_quartile'] == 1].sort_values('Frequency', ascending=False).head(10)


# In[42]:


loyal_ones = rfm_table_copy[(rfm_table_copy['f_quartile'] == 1) & (rfm_table_copy['cust_type'] == 'NA')] #['cust_type'].replace('NA', 'Loyal_Cust')
rfm_table_copy['cust_type'][loyal_ones.index] = 'Loyal_Cust'


# ### (4) Loyal Joes
# 
# - Loyal Joes buy often, but don’t spend very much.
# - Goal should be to increase the share of wallet you have from this customer.
# - Send offers that require them to “Spend 100 rupees to save 20 rupees” and “Buy 4, Get 1 Free.” These offers create high hurdles that must be cleared to gain the reward, and will increase the amount these loyal customers spend with you.

# In[43]:


rfm_table_copy[(rfm_table_copy['f_quartile'] == 1) & (rfm_table_copy['m_quartile'] == 4)].head()


# In[44]:


loyal_joes = rfm_table_copy[(rfm_table_copy['f_quartile'] == 1) & (rfm_table_copy['m_quartile'] == 4) & (rfm_table_copy['cust_type'] == 'Loyal_Cust')]['cust_type'].replace('Loyal_Cust', 'Loyal_Joes')
rfm_table_copy['cust_type'][loyal_joes.index] = 'Loyal_Joes'


# ### (5) New Comers
# 
# - New Spenders are new customers that spent a lot of money on their first order(s). This is the kind of customer you want to convert into a loyal, regular customer that loves your products and brand. Be sure to welcome them and thank them for making a first purchase, and follow it up with unique incentives to come back again.
# - Consider branding the email with a special note from the CEO, and include a survey to ask about their experience

# In[45]:


rfm_table_copy[rfm_table_copy['RFMscore'] == '141'].sort_values(by='Monetary', ascending=False).head(10)


# In[46]:


new_comers = rfm_table_copy[(rfm_table_copy['RFMscore'] == '141') & (rfm_table_copy['cust_type'] == 'Big_Spenders')]
rfm_table_copy['cust_type'] [new_comers.index] = 'New_comers'


# ### (6) Lost Customers
# 
# - Lost Customers used to buy frequently from you, and at one point they spent a lot with you, but they’ve stopped. Now it’s time to win them back.
# - They might be lost to a competitor; they might not have need of your products anymore, or they might have had a bad customer service experience with you.

# In[47]:


rfm_table_copy[rfm_table_copy['RFMscore'] == '411'].sort_values(by='Monetary', ascending=False).head(10)


# In[48]:


lost_cust = rfm_table_copy[(rfm_table_copy['RFMscore'] == '411') & (rfm_table_copy['cust_type'] == 'Big_Spenders')]
rfm_table_copy['cust_type'] [lost_cust.index] = 'Lost_cust'


# ### (7) Almost Lost Customers
# 
# - It has just been less time since they purchased. These customers might warrant more aggressive discounts so that you can win them back before it’s too late.
# - It is much less expensive it is to keep customers compared to winning new ones

# In[49]:


rfm_table_copy[rfm_table_copy['RFMscore'] == '311'].sort_values(by='Monetary', ascending=False).head(10)


# In[50]:


almost_lost_cust = rfm_table_copy[(rfm_table_copy['RFMscore'] == '311') & (rfm_table_copy['cust_type'] == 'Big_Spenders')]
rfm_table_copy['cust_type'] [almost_lost_cust.index] = 'Almost_lost_cust'


# ### (8) Splurgers
# 
# - Splurgers combine a high Monetary Value with a low Frequency, which means they’ve spent a lot of money in just a few orders. Because they have the wealth and willingness to spend a lot with you, target high priced products with good margins at this group. 
# - This group might also correspond with seasonal events or even just the typical buying cycle of your product’s wear.

# In[51]:


rfm_table_copy[(rfm_table_copy['f_quartile'] == 4) & (rfm_table_copy['m_quartile'] == 1)].head(10)


# In[52]:


splurgers = rfm_table_copy[(rfm_table_copy['RFMscore'].apply(lambda x : x[-2:] == '41')) & (rfm_table_copy['cust_type'] == 'Big_Spenders')]
rfm_table_copy['cust_type'] [splurgers.index] = 'Splurgers'


# ### (10) Deadbeats
# 
# - These customers spent very little, bought very few times, and last ordered quite a while ago. They are unlikely to be worth much time, so put them in your general house list and consider a re-opt-in campaign.

# In[53]:


rfm_table_copy[rfm_table_copy['RFMscore'] == '444'].sort_values(by='Monetary', ascending=False).head(10)


# In[54]:


deadbeats = rfm_table_copy[rfm_table_copy['RFMscore'] == '444']
rfm_table_copy['cust_type'][deadbeats.index] = 'Deadbeats'


# In[ ]:


#replacing NA wit Other

rfm_table_copy['cust_type'].replace('NA', 'Other', inplace=True)


# ### Viz. depicting Count of different customers & their impact on Monetary value

# In[56]:


sns.set_context('notebook')
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.figure(figsize=(15,15))

plt.subplot(2,1,1)
sns.countplot(rfm_table_copy['cust_type'], palette='inferno')
plt.xticks(rotation=0)
plt.title('Customer Types')

plt.subplot(2,1,2)
sns.swarmplot(rfm_table_copy['cust_type'], rfm_table_copy['Monetary'], palette='inferno')
plt.xticks(rotation=0)
plt.title('Impact of Customer Types on Monetary')


# - We can see that "others" contribute more to the our grouping.
# - While, Best Customers and Big spenders are having huge impact on Monetary value. 

# ### TreeMap

# In[57]:


get_ipython().system('pip install squarify')


# In[58]:


import squarify

plt.figure(figsize = (10,30))
regions = rfm_table_copy['cust_type'].value_counts().to_frame()
regions['count_val']=regions.values
regions['percent']=regions['count_val']/(regions.values.sum())*100


# In[59]:


regions


# In[60]:


sns.set_context('paper')
plt.figure(figsize=(25,18))
squarify.plot(sizes = regions["cust_type"].values, label = (regions.index+'-'+regions.count_val.astype(str)+'\n'+regions.percent.round(2).astype(str))+'%',
              color = sns.color_palette("gist_stern", 10), alpha = 0.9)
plt.xticks([])
plt.yticks([])
plt.title("Treemap of Main Category", fontsize = 18)


# - The Above treemap shows how specific customer segement is contributing to us.
# - By analysing the map, we can make targeted offers, discounts, etc on specific group of customers. That will help us further analyze how they are adding value to our business.

# In[61]:


#dropping examples with 0 monetary

rfm_table_copy[rfm_table_copy['Monetary'] == 0]


# In[ ]:


rfm_table_copy.drop(rfm_table_copy.index[314], inplace=True)


# ### Log Transformation & Normalization

# In[63]:


#Log transformation

rfm_table_log = np.log1p(rfm_table_copy[['Recency', 'Frequency', 'Monetary']])
rfm_table_log = pd.concat([rfm_table_copy['CustomerID'], rfm_table_log], axis=1)
rfm_table_log.set_index(['CustomerID'], inplace=True)
rfm_table_log.head(10)


# - We need to do Log Transformation since values of each feature has outliers. And Outliers can affect our clustering process.

# In[64]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

cols = rfm_table_log.columns

norm_df = scaler.fit_transform(rfm_table_log[['Recency', 'Frequency', 'Monetary']])
norm_df = pd.DataFrame(data=norm_df, columns=cols, index=rfm_table_log.index)
norm_df.head(10)


# - We need to Scale our data and we will use Sklearn's StandardScaler for that.

# In[65]:


sns.set_context('paper')
plt.subplots_adjust(hspace=1, wspace=1)
plt.figure(figsize=(20,15))

plt.subplot(3,1,1)
sns.distplot(norm_df['Recency'], color='r')

plt.subplot(3,1,2)
sns.distplot(norm_df['Frequency'], color='g')

plt.subplot(3,1,3)
sns.distplot(norm_df['Monetary'], color='b')


# - Above is a viz. of distribution of R, F & M after Log Transformation & Scaling.

# ### K-Means Clustering

# - We'll Initialize K-means with K clusters.
# - By Intializing K-means we'll get **Sum of Squared Errors (SSE).**
# - SSE is the sum of the squared differences between each observation and its group's mean.

# In[ ]:


from sklearn.cluster import KMeans, MiniBatchKMeans

k = np.arange(1,15)
sse = []

for i in k:
    model = MiniBatchKMeans(i)
    model.fit(norm_df)
    sse.append(model.inertia_/100)


# In[67]:


cluster = pd.DataFrame({"Number of Clusters": k , "Error":sse})
cluster


# In[68]:


plt.figure(figsize=(10,7))
plt.plot(cluster['Number of Clusters'], cluster['Error'], marker="o", linestyle="--" , c="black")
plt.title("Elbow plot")


# - The above **Elbow Plot** elaborates optimum number of clusters.
# - We can consider **4** as we see after 4 the plot is quite flat with mimimal change / variation.
# - Now we'll initialize K-means on 4 clusters and iterate for 200 times.

# In[ ]:


kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=200, n_init=10)
kmeans.fit(norm_df)
labels = kmeans.labels_


# In[70]:


rfm_k4 = rfm_table_copy.assign(Cluster=labels)

grouped_rfm = rfm_k4.groupby('Cluster').agg({'Recency':'mean',
                                             'Frequency': 'mean',
                                             'Monetary': ['mean','count']}).round(2)

grouped_rfm


# - Above we had assigned Cluster labels to our RFM table.
# - Then we group by cluster and aggregate to find Average of R, F & M for each cluster along with its Size.

# In[71]:


sns.set_context('notebook')
plt.subplots_adjust(hspace=0.5, wspace=0.7)
plt.figure(figsize=(23,5))

plt.subplot(1,3,1)
sns.boxenplot(rfm_k4['Cluster'], rfm_k4['Monetary'])
plt.xticks(ticks=[0,1,2,3], labels=['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3'], rotation=90)
plt.title('Impact of Specific Cluster on Monetary')

plt.subplot(1,3,2)
sns.boxenplot(rfm_k4['Cluster'], rfm_k4['Recency'])
plt.xticks(ticks=[0,1,2,3], labels=['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3'], rotation=90)
plt.title('Impact of Specific Cluster on Recency')

plt.subplot(1,3,3)
sns.boxenplot(rfm_k4['Cluster'], rfm_k4['Frequency'])
plt.xticks(ticks=[0,1,2,3], labels=['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3'], rotation=90)
plt.title('Impact of Specific Cluster on Frequency')


# #### Observations :
# 
# 
#  - **Cluster 3** is highly affecting our Monetary value and Frequency and also they are recent, we can say the Cluster 3 contains our **Big Spenders** to which we can target expensive products.
# 
# 
#  - **Cluster 2** is decent, its affect on Monetary value is good. We can provide them reasonable discounts and offers to keep them involved in purchase activity. Most of the customers have bad Recency so we can try to get them again before we tend to lose them. Although, They are our **Best Customers.**
# 
# 
#  - **Cluster 1** is interesting. Their recency varies from best to bad. They don't spend much. They are combination of **Loyal Joes, Almost Lost Customers & Lost Customers**. We can try to give them good discounts / attractive offers pushing them to spend more.
# 
# 
#  - **Cluster 0** contains our **Loyal Customers** their recency is very nice, they don't spend much so we'll have to provide them good discounts & attractive offers.

# In[72]:


rfm_table_log = rfm_table_log.assign(Cluster=labels)
rfm_table_log.head()


# - Above table helps us find out which **Customer** belongs to which **Cluster.**

# In[117]:


import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
for c, m, zl, zh in [('#CD9575', 'o', 0, 0), ('#EA7E5D', '^', 0, 0), ('#FC2847', '+', 0, 0)]:
    xs = rfm_table_log[rfm_table_log['Cluster'] == 3]['Recency']
    ys = rfm_table_log[rfm_table_log['Cluster'] == 3]['Frequency']
    zs = rfm_table_log[rfm_table_log['Cluster'] == 3]['Monetary']
    #ax.scatter(xs, ys, zs, marker=m, depthshade=True, c=c)
    ax.scatter(xs, ys, zs, marker=m, c=c)
    
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
ax.view_init(65, 35)
plt.show()


# ***By this we conclude our RFM Analysis. We cannot neglect any cluster because they are our source of revenue. We must focus on providing them best offers / discounts to Cluster 0, Cluster 1 and Cluster 2. While Cluster 3 has big spenders to which we can target marketing our Top of the Line (new) product(s). While to Cluster 0, Cluster 1 and Cluster 2 we must provides "Buy x units, get y units free" offers to let them spend more.***
