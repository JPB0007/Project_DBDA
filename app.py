# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 00:31:28 2022

@author: bhoya
"""

#import pandas as pd
# =============================================================================
# import numpy as np
# import streamlit as st
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.ensemble import RandomForestClassifier
#from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import accuracy_score
# from sklearn.feature_selection import RFE
# from statsmodels.stats.outliers_influence import variance_inflation_factor
#from sklearn.metrics import precision_score 
#from sklearn.metrics import recall_score
#from sklearn.metrics import f1_score
# =============================================================================
import pickle

#scaler = StandardScaler()

#loading the model
model = pickle.load(open('random_forest_model.h5','rb'))

#loading the test data
test_X = pickle.load(open('test_data.h5','rb'))

#loading the target for evaluating model
y_true = pickle.load(open('target.h5','rb'))

#making prediction bassed on credentials
y_pred = model.predict(test_X)

#making a list of predictions
y_pred_list = model.predict(test_X).tolist()

#initializing the dictionary with name 'd'
d = dict()

#creating a dictionary of customer key and respective predictions 
for i in range(len(y_pred_list)):
    d[i] = y_pred_list[i]

#printing id and prediction
inp = int(input("Enter ID here: "))
prediction = d.get(inp)

if prediction == 0:
    def_nonDef = "Defaulter"
elif prediction == 1:
    def_nonDef = "Non-Defaulter"
print(f"Based on the credentials the customer with id [{inp}] is more likely to be a [{def_nonDef}].")


# =============================================================================
# acc = accuracy_score(y_true, y_pred) 
# prec = precision_score(y_true, y_pred) 
# rec = recall_score(y_true, y_pred)
# f1 = f1_score(y_true, y_pred)
# 
# print("Accuracy: ",acc)
# print("Precision: ",prec)
# print("Recall: ",rec)
# print("F1_Score: ",f1)
# =============================================================================

        


