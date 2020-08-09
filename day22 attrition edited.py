# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 20:28:53 2020

@author: Aparna s nair

"""
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm

label_encoder=preprocessing.LabelEncoder()
dataset=pd.read_csv("Attrition data set.csv")

#setting target
dataset["Attrition"]=label_encoder.fit_transform(dataset["Attrition"])
y1=dataset.Attrition


#changing strings to numerical values
dataset["EducationField"]=label_encoder.fit_transform(dataset["EducationField"])
dataset["Gender"]=label_encoder.fit_transform(dataset["Gender"])
dataset["BusinessTravel"]=label_encoder.fit_transform(dataset["BusinessTravel"])
dataset["Department"]=label_encoder.fit_transform(dataset["Department"])
dataset["JobRole"]=label_encoder.fit_transform(dataset["JobRole"])
dataset["MaritalStatus"]=label_encoder.fit_transform(dataset["MaritalStatus"])

#dealing with Nan data
dataset['NumCompaniesWorked'].fillna(dataset['NumCompaniesWorked'].mean(), inplace=True)
dataset['TotalWorkingYears'].fillna(dataset['TotalWorkingYears'].mean(), inplace=True)
x2=dataset[["Age","BusinessTravel","Department","DistanceFromHome","Education","EducationField","Gender","JobLevel","JobRole","MaritalStatus","MonthlyIncome","NumCompaniesWorked","PercentSalaryHike","StandardHours","StockOptionLevel","TotalWorkingYears","TrainingTimesLastYear","YearsAtCompany","YearsSinceLastPromotion","YearsWithCurrManager"]]

x3=sm.add_constant(x2)

logistics=sm.Logit(y1,x3)
result1=logistics.fit()

print(result1.summary())
print(' INFERENCE')
print("1)Factors such as Business travel,distance from home,job level,standard hours,years at company have Pvalue above 0.5,so they are not significant factors affecting Attrition.")
print("2)Rest all factors have Pvalue less than 0.5, implies they all are significant factors affecting Attrition.")