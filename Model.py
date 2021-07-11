# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 15:22:58 2021

@author: Pradhan
"""
#importing necesary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from imblearn.combine import SMOTEENN
%matplotlib inline

churn_data = pd.read_csv('Telco_Customer_Churn_Data.csv')
churn_data
# Exploring Data and feature engineering.
churn_data.head()
churn_data.TotalCharges = pd.to_numeric(churn_data.TotalCharges, errors='coerce')
churn_data.dropna(how = 'any', inplace=True)
labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
churn_data['tenure_group'] = pd.cut(churn_data.tenure, range(1, 80, 12), right=False, labels=labels)
churn_data.drop(columns = ['customerID','tenure'],axis=1,inplace=True)
churn_data['Churn']= np.where(churn_data.Churn=='Yes',1,0)
churn_data_converted = pd.get_dummies(churn_data)
churn_data_converted.head()
churn_data_converted.shape
#rsetting index
churn_data_converted= churn_data_converted.reset_index(drop=True)
#splitting the data
X = churn_data_converted.drop('Churn',axis=1)
X

Y = churn_data_converted['Churn']
Y

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
#fitting Decision Tree Classifier
model_dt=DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=6, min_samples_leaf=8)
model_dt.fit(x_train,y_train)
y_pred=model_dt.predict(x_test)
y_pred

model_dt.score(x_test,y_test)


print(classification_report(y_test, y_pred, labels=[0,1]))

# handling imbalanced data using upsampling.
sm = SMOTEENN()
X_resampled, y_resampled = sm.fit_resample(X,Y)
xr_train,xr_test,yr_train,yr_test=train_test_split(X_resampled, y_resampled,test_size=0.2)
model_dt_smote=DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=6, min_samples_leaf=8)
model_dt_smote.fit(xr_train,yr_train)
yr_predict = model_dt_smote.predict(xr_test)
model_score_r = model_dt_smote.score(xr_test, yr_test)
print(model_score_r)
print(metrics.classification_report(yr_test, yr_predict))

print(metrics.confusion_matrix(yr_test, yr_predict))
# fitting randomforestclassifier
from sklearn.ensemble import RandomForestClassifier
model_rf=RandomForestClassifier(n_estimators=100, criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)
model_rf.fit(x_train,y_train)
y_pred=model_rf.predict(x_test)
model_rf.score(x_test,y_test)


print(classification_report(y_test, y_pred, labels=[0,1]))

sm = SMOTEENN()
X_resampled1, y_resampled1 = sm.fit_resample(X,Y)
xr_train1,xr_test1,yr_train1,yr_test1=train_test_split(X_resampled1, y_resampled1,test_size=0.2)
model_rf_smote=RandomForestClassifier(n_estimators=100, criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)
model_rf_smote.fit(xr_train1,yr_train1)
yr_predict1 = model_rf_smote.predict(xr_test1)
model_score_r1 = model_rf_smote.score(xr_test1, yr_test1)
print(model_score_r1)
print(metrics.classification_report(yr_test1, yr_predict1))


print(metrics.confusion_matrix(yr_test1, yr_predict1))
# handling data using PCA 
from sklearn.decomposition import PCA
pca = PCA(0.9)
xr_train_pca = pca.fit_transform(xr_train1)
xr_test_pca = pca.transform(xr_test1)
explained_variance = pca.explained_variance_ratio_
model=RandomForestClassifier(n_estimators=100, criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)
model.fit(xr_train_pca,yr_train1)
yr_predict_pca = model.predict(xr_test_pca)
model_score_r_pca = model.score(xr_test_pca, yr_test1)
print(model_score_r_pca)
print(metrics.classification_report(yr_test1, yr_predict_pca))


import pickle
filename = 'model.pkl'
pickle.dump(model_rf_smote, open(filename, 'wb'))
load_model = pickle.load(open(filename, 'rb'))
model_score_r1 = load_model.score(xr_test1, yr_test1)
model_score_r1




