#!/usr/bin/env python
# coding: utf-8

# # Walmart Sales Analysis using Python

# ## Importing libraries
# 

# In[5]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[6]:


df = pd.read_csv('D:/Python/Walmart_Store_sales.csv')


# In[38]:


df.info()


# In[7]:


df.describe()


# In[42]:


len(df.columns)


# In[8]:


#missing values
miss_values=df.isna().sum().sort_values(ascending=False)
miss_values


# In[9]:


df.Store.value_counts().head(10)


# In[13]:


df1 = df.groupby('Store')['Weekly_Sales'].sum().reset_index(name= 'sales').sort_values(by = 'sales', ascending = False).head(5)
df1


# In[15]:


type(df1)
df1


# In[16]:


df.Weekly_Sales.sort_values(ascending = False).reset_index(name = "max_sales").head(5)


# In[92]:


#Store 20 has maximum weekly sales


# In[17]:


df2 = df.groupby('Store').std()


# In[18]:


df2.Weekly_Sales.sort_values(ascending = False).head(5)


# In[ ]:


#Store 14 has maximum standard deviation.


# In[ ]:





# ### Converting Date field into date format

# In[19]:


df['Date'] = pd.to_datetime(df['Date'])


# In[20]:


type(df.Date)


# In[21]:


df['Year'] = df['Date'].dt.year


# In[22]:


df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day


# In[23]:


df['Quarter'] = df['Date'].dt.quarter


# In[24]:


df3=df.groupby(['Year','Quarter']).sum()


# In[25]:


df


# ### Sales Data for the year 2012

# In[26]:



Sales_2012 = df[df['Year']== 2012]


# In[27]:


Sales_2012


# ### Quaterly_sales for the year 2012
# 

# In[28]:


sales_2012_by_quarter = Sales_2012.groupby(['Store','Quarter'])['Weekly_Sales'].sum().reset_index(name = 'Quaterly_sales')
sales_2012_by_quarter = pd.DataFrame(sales_2012_by_quarter)


# In[226]:


sales_2012_by_quarter


# ## Calculating Growth Rate
# ### Growth rate = (present value - past value)/past value
# ### growth rate for Q2 sales = (Q2-Q1)/Q1
# 

# In[29]:




Q = sales_2012_by_quarter.pivot_table(index = 'Store',columns='Quarter',values='Quaterly_sales')


# In[39]:


type(Q)
Q
Q['GR_Q2'] = (Q[2]-Q[1])*100/Q[1]


# In[50]:



Q2_growth = Q['GR_Q2'].sort_values(ascending = False).head(10)

Q2_growth


# In[54]:


plt.figure(figsize=(10,10))
ax=sns.barplot(Q2_growth.index, Q2_growth.values,).set(title = "Q2 - 2012 Growth Rate ")
ax.set_xlabel("Store")
ax.set_ylabel("Growth Rate")


# In[52]:


#Store 17 has maximum growth in Q2 of 2012.


# In[55]:


df


# ### Holiday Sales

# In[98]:


df3 = pd.read_csv('D:/Python/Walmart_Store_sales.csv')
df3 = df3[df3['Holiday_Flag']==1]


# In[ ]:


#Super Bowl: 12-Feb-10, 11-Feb-11, 10-Feb-12, 8-Feb-13
#Labour Day: 10-Sep-10, 9-Sep-11, 7-Sep-12, 6-Sep-13
#Thanksgiving: 26-Nov-10, 25-Nov-11, 23-Nov-12, 29-Nov-13
#Christmas: 31-Dec-10, 30-Dec-11, 28-Dec-12, 27-Dec-13


# In[113]:


def holiday(x):
    if x['Date'] in  ['12-02-2010','11-02-2011','10-02-2012','08-02-2013']: return "Super Bowl Day"
    elif x['Date'] in  ['10-09-2010','09-09-2011','07-09-2012','06-09-2013']:  return "Labour Day"
    elif x['Date'] in  ['26-11-2010','25-11-2011','23-11-2012','29-11-2013']:  return "ThanksGiving Day"
    elif x['Date'] in  ['31-12-2010','30-12-2011','28-12-2012','27-12-2013']:  return "Christmas Day"
    else: return None
    
df3['Holiday_name'] = df3.apply(holiday, axis = 1)    
df3


# In[104]:



df1['Holiday_name'] = df3.apply(holiday, axis = 1)
df1


# In[114]:


#holiday week sales
Holiday_sales = df3['Holiday_name'].value_counts()
Holiday_sales


# In[127]:


plt.figure(figsize=(7,7))
ax=sns.barplot(Holiday_sales.index, Holiday_sales.values,).set(title = "Holiday Sales ")
ax.set_xlabel("Holiday Name")
ax.set_ylabel("Count")


# In[118]:


data = df3[['Holiday_name','Weekly_Sales']]


# In[128]:


plt.figure(figsize=(7,7))
ax=sns.barplot(x= 'Holiday_name', y= 'Weekly_Sales',data= data).set(title = 'Holiday week Sales')


# In[168]:


#Weekly_sales is maximum on 'ThanksGiving Day' 
df


# In[173]:


#Monthly sales


# In[175]:


Monthly_sales = df1.groupby('Month')['Weekly_Sales'].sum().reset_index(name= 'sales').sort_values(by = 'sales', ascending = False).head()


# In[176]:


Monthly_sales


# In[149]:


plt.figure(figsize=(7,7))
ax=sns.barplot(x= 'Month', y= 'sales',data= Monthly_sales).set(title = 'Monthly Sales')


# In[3]:


#Non Holiday sales data
Non_holiday = df[df['Holiday_Flag']==0]
Non_holiday


# In[186]:


Non_holiday['NH_mean'] =Non_holiday.groupby('Store')['Weekly_Sales'].mean()
Non_holiday['NH_mean']


# ### Non holiday sales mean

# In[2]:



Non_holidayMean = Non_holiday.groupby("Store")["Weekly_Sales"].mean().reset_index(name="Average Sale").sort_values(by='Average Sale', ascending=False)


# In[1]:


Non_holidayMean['Average Sale'] = Non_holidayMean['Average Sale'].round(3)


# In[256]:


store1.columns


# In[259]:


#Store 1 data - Prediction model to forecast demand

store1 = df[df['Store']==1]
store1= store1.drop(['Store','Year', 'Month', 'Day', 'Quarter',
       'Holiday_name'],axis =1)
store1.shape
store1.columns


# In[260]:


ax = sns.heatmap(store1.corr().round(2), annot = True)


# In[262]:


#features and target
x = store1.drop(['Weekly_Sales','Date'], axis = 1)
y = store1[['Weekly_Sales']]


# In[263]:


#Building linear regression model
#training
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = .8, random_state = 10)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[234]:


from sklearn.linear_model import LinearRegression
lreg = LinearRegression()


# In[240]:


lreg.fit(x_train, y_train )


# In[242]:


lreg.coef_


# In[243]:


lreg.intercept_


# In[247]:


y_pred = lreg.predict(x_test)


# In[248]:


# Mean_Square_Error
from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y_test, y_pred)
MSE


# In[249]:


#Root Mean square Error
np.sqrt(MSE)

