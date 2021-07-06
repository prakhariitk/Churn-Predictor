# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 15:06:35 2021

@author: LENOVO
"""
import pandas as pd
churn_data = pd.read_csv('Telco_Customer_Churn_Data.csv')
churn_data

churn_data.head()
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
%matplotlib inline
churn_data.columns.values
churn_data.dtypes
churn_data.describe()
churn_data['Churn'].value_counts()/ len(churn_data['Churn'])
churn_data['Churn'].value_counts()
churn_data.info(verbose=True)
churn_data.TotalCharges = pd.to_numeric(churn_data.TotalCharges, errors='coerce')
churn_data.isnull().sum()
churn_data.loc[churn_data['TotalCharges'].isnull()==True]
churn_data.dropna(how = 'any', inplace=True)
churn_data.drop(columns = ['customerID','tenure'],axis=1,inplace=True)
churn_data.head()
churn_data_copy = churn_data.copy()
for i, predictor in enumerate(churn_data.drop(columns=['Churn','TotalCharges','MonthlyCharges'])):
  plt.figure(i)
  sns.countplot(data=churn_data,x=predictor,hue='Churn')
churn_data['Churn']= np.where(churn_data.Churn=='Yes',1,0)
churn_data.head()
churn_data_converted = pd.get_dummies(churn_data)
churn_data_converted.head()
churn_data_converted.shape
sns.lmplot(data=churn_data_converted, x='MonthlyCharges',y='TotalCharges',fit_reg=False)
Mth = sns.kdeplot(churn_data_converted.MonthlyCharges[(churn_data_converted["Churn"] == 0) ],
                color="Red", shade = True)
Mth = sns.kdeplot(churn_data_converted.MonthlyCharges[(churn_data_converted["Churn"] == 1) ],
                ax =Mth, color="Blue", shade= True)
Mth.legend(["No Churn","Churn"],loc='upper right')
Mth.set_ylabel('Density')
Mth.set_xlabel('Monthly Charges')
Mth.set_title('Monthly charges by churn')

Tot = sns.kdeplot(churn_data_converted.TotalCharges[(churn_data_converted["Churn"] == 0) ],
                color="Red", shade = True)
Tot = sns.kdeplot(churn_data_converted.TotalCharges[(churn_data_converted["Churn"] == 1) ],
                ax =Tot, color="Blue", shade= True)
Tot.legend(["No Churn","Churn"],loc='upper right')
Tot.set_ylabel('Density')
Tot.set_xlabel('Total Charges')
Tot.set_title('Total charges by churn')

plt.figure(figsize=(20,8))
churn_data_converted.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')


plt.figure(figsize=(15,15))
sns.heatmap(churn_data_converted.corr(), cmap="Paired")

zero_churn_df=churn_data.loc[churn_data["Churn"]==0]
one_churn_df=churn_data.loc[churn_data["Churn"]==1]


def uniplot(df,col,title,hue =None):
    
    sns.set_style('whitegrid')
    sns.set_context('talk')
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.titlepad'] = 30
    
    
    temp = pd.Series(data = hue)
    fig, ax = plt.subplots()
    width = len(df[col].unique()) + 7 + 4*len(temp.unique())
    fig.set_size_inches(width , 8)
    plt.xticks(rotation=45)
    plt.yscale('log')
    plt.title(title)
    ax = sns.countplot(data = df, x= col, order=df[col].value_counts().index,hue = hue,palette='bright') 
        
    plt.show()
    

uniplot(one_churn_df,col='Partner',title='Distribution of Gender for Churned Customers',hue='gender') 

uniplot(one_churn_df,col='PaymentMethod',title='Distribution of Gender for Churned Customers',hue='gender')

uniplot(one_churn_df,col='Contract',title='Distribution of Gender for Churned Customers',hue='gender')






























   