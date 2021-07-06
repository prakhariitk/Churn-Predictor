# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 20:53:34 2021

@author: LENOVO
"""

import numpy as np
import pickle
import pandas as pd
import streamlit as st
from PIL import Image

def welcome():
    return "welcome all"
def predict(gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,
            MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection
            ,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,
            PaymentMethod,MonthlyCharges,TotalCharges):
    SeniorCitizen = int(SeniorCitizen)
    tenure = int(tenure)
    MonthlyCharges = float(MonthlyCharges)
    TotalCharges = float(TotalCharges)
    cols = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    df_1 = pd.read_csv('first_telc.csv',usecols=cols)
    df_1.TotalCharges = pd.to_numeric(df_1.TotalCharges, errors='coerce')
    model = pickle.load(open("model.pkl", "rb"))
    data = [[gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,
             MultipleLines,InternetService,OnlineSecurity,OnlineBackup,
             DeviceProtection,TechSupport,StreamingTV,StreamingMovies,
             Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges]]
    new_df = pd.DataFrame(data, columns = ['gender','SeniorCitizen','Partner','Dependents'
                                           ,'tenure','PhoneService','MultipleLines','InternetService',
                                           'OnlineSecurity','OnlineBackup','DeviceProtection'
                                           ,'TechSupport','StreamingTV','StreamingMovies',
                                           'Contract','PaperlessBilling','PaymentMethod',
                                           'MonthlyCharges','TotalCharges'])
    
    df_2 = pd.concat([df_1, new_df], ignore_index = True) 
    df_2.dropna(how = 'any', inplace=True)
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    df_2['tenure_group'] = pd.cut(df_2.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)
    df_2.drop(columns= ['tenure'], axis=1, inplace=True) 
    
    new_df__dummies = pd.get_dummies(df_2)
    probablity = model.predict_proba(new_df__dummies.tail(1))[:,1]
    
    return probablity


def main():
    st.title("Telco_Churn")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Telco_Churn</h2>
    </div>
    """ 
    st.markdown(html_temp,unsafe_allow_html=True)
    gender = st.text_input("gender","Type Here")
    SeniorCitizen = st.text_input("SeniorCitizen","Type Here")
    Partner = st.text_input("Partner","Type Here")
    Dependents = st.text_input("Dependents","Type Here")
    tenure = st.text_input("tenure","Type Here")
    PhoneService = st.text_input("PhoneService","Type Here")
    MultipleLines = st.text_input("MultipleLines","Type Here")
    InternetService = st.text_input("InternetService","Type Here")
    OnlineSecurity = st.text_input("OnlineSecurity","Type Here")
    OnlineBackup = st.text_input("OnlineBackup","Type Here")
    DeviceProtection = st.text_input("DeviceProtection","Type Here")
    TechSupport = st.text_input("TechSupport","Type Here")    
    StreamingTV = st.text_input("StreamingTV","Type Here")
    StreamingMovies = st.text_input("StreamingMovies","Type Here")
    Contract = st.text_input("Contract","Type Here")
    PaperlessBilling = st.text_input("PaperlessBilling","Type Here")
    PaymentMethod = st.text_input("PaymentMethod","Type Here")     
    MonthlyCharges = st.text_input("MonthlyCharges","Type Here")
    TotalCharges = st.text_input("TotalCharges","Type Here")  
    result = ""
    if st.button("predict"):
        result = predict(gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges)
    st.success('output is {}'.format(result))    
    
if __name__=='__main__':
    main()
    