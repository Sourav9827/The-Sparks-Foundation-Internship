# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 10:27:07 2021

@author: sRv
"""

# Importing all libraries required in this notebook
import os
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns 
from statsmodels.formula.api import ols
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

## Reading data from remote link
#url = "http://bit.ly/w-data"
#s_data = pd.read_csv(url)
#print("Data imported successfully")

## Saving The Data For Offline Use
#s_data.to_csv(r"D:\Temp\studentsinfo.csv", index = False)

#Loding Data
os.chdir("D:\Temp")
s_data = pd.read_csv('studentsinfo.csv')

# Data Visualsation
plt.scatter(x='Hours', y='Scores',data=s_data)
plt.xlabel("hours")
plt.ylabel("Scores")
plt.show()

# Ordinary Least Square Estimation
lm = ols("Scores ~ Hours", s_data).fit()
lm.summary()

#Preparing The Data
x_train, x_test, y_train, y_test = train_test_split(s_data.drop(columns='Scores'), s_data['Scores'], test_size=0.2, random_state=3)
print("\n\n Shapes Of Training And Test Data:\n",x_train.shape, x_test.shape, y_train.shape, y_test.shape)
base_pred = np.mean(y_test)
print("Base Prediction:\t",base_pred)
base_pred = np.repeat(base_pred, len(y_test))
base_root_mean_square_error = np.sqrt(mean_squared_error(y_test, base_pred))
print("RMSE:\t",base_root_mean_square_error)

#Linear Regression

lnr = LinearRegression(fit_intercept=True)
lnr_model = lnr.fit(x_train, y_train)
bi = lnr_model.coef_
b0 = lnr_model.intercept_
prediction_lnr = lnr.predict(x_test)

df_lnr = pd.DataFrame({'Actual': y_test, 'Predicted': prediction_lnr})

lnr_model_mse = mean_squared_error(y_test, prediction_lnr)
lnr_model_rmse = np.sqrt(lnr_model_mse)
print("\n\n RMSE Of Linear Regression Model:\t",lnr_model_rmse)

r2_lnr_test = lnr_model.score(x_test, y_test)
print("\n R-Squared value Of Test Data: \t",r2_lnr_test)
r2_lnr_train = lnr_model.score(x_train, y_train)
print("\n R Squared Value For Train Data Are: \t", r2_lnr_train)

lnr_residuals=y_test-prediction_lnr
sns.regplot(prediction_lnr, lnr_residuals, scatter=True, fit_reg=True)
plt.title("Residuals For Decision Linear Regression Model")
plt.show()

#Decision Tree

tree = DecisionTreeRegressor()
model_tree = tree.fit(x_train, y_train)
prediction_tree = tree.predict(x_test)

df_tree = pd.DataFrame({'Actual': y_test, 'Predicted': prediction_tree})

tree_model_mse = mean_squared_error(y_test, prediction_tree)
tree_model_rmse = np.sqrt(tree_model_mse)
print("\n\n RMSE Of Decision Tree Model:\t",tree_model_rmse)

r2_tree_test = model_tree.score(x_test, y_test)
print("\n R-Squared value Of Test Data: \t",r2_tree_test)
r2_tree_train = model_tree.score(x_train, y_train)
print("\n R-Squared value Of Train Data: \t", r2_tree_train)

residuals_tree=y_test-prediction_tree
sns.regplot(x=prediction_tree, y=residuals_tree, scatter=True, fit_reg=True)
plt.title("Residuals For Decision Tree Model")
plt.show()

#Random Forest 

rf = RandomForestRegressor(random_state=10)
model_rf = rf.fit(x_train, y_train)
prediction_rf = rf.predict(x_test)

df_rf = pd.DataFrame({'Actual': y_test, 'Predicted': prediction_rf})

rf_mse = mean_squared_error(y_test, prediction_rf)
rf_rmse = np.sqrt(rf_mse)
print("\n\n RMSE Of Random Forest Model: \t",rf_rmse)

r2_rf_test = model_rf.score(x_test, y_test)
print("\n R-Squared value Of Test Data: \t",r2_rf_test)
r2_rf_train = model_rf.score(x_train, y_train)
print("\n R-Squared value Of Test Data: \t", r2_rf_train)

residuals_rf=y_test-prediction_rf
sns.regplot(x=prediction_rf, y=residuals_rf, scatter=True, fit_reg=True)
plt.title("Residuals For Random Forest Model")
plt.show()