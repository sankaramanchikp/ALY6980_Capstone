# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 15:39:00 2021

@author: praka
"""

import pandas as pd
import numpy as np
from datetime import date
from pycaret.classification import *

##Reading the dataset into Pandas Dataframe
LSH_Data = pd.read_csv('NE_LSH.csv')

#Convert object columns to Numeric
for i in range(0, len(LSH_Data.columns)):
    LSH_Data.iloc[:,i] = pd.to_numeric(LSH_Data.iloc[:,i], errors='ignore')
    # errors='ignore' lets strings remain as 'non-null objects'

#Convert columns with % into Numeric
LSH_Data['MB % hemoglobin subunit alpha glycation'] = pd.to_numeric(LSH_Data['MB % hemoglobin subunit alpha glycation']
                                                                    .str.replace('%', ''))
LSH_Data['MB % hemoglobin subunit beta glycation'] = pd.to_numeric(LSH_Data['MB % hemoglobin subunit beta glycation']
                                                                   .str.replace('%', ''))
LSH_Data['fingerprick glycation HbB'] = pd.to_numeric(LSH_Data['fingerprick glycation HbB'].str.replace('%', ''))
LSH_Data['fingerprick glycation HbA'] = pd.to_numeric(LSH_Data['fingerprick glycation HbA'].str.replace('%', ''))

##handle missing values
##Remove column BC dose & Method as it has a lot of missing values and of less significance

LSH_Data = LSH_Data.drop(columns=['BC dose & method','Control - Authenticated HbA1c 1 glucose = to 5.88% glycated hemoglobin alpha',
                                  'Control - Authenticated HbA1c #of glucose = to 96% glycated hemoglobin beta',
                                  'Control HbB glycan modification lysine site'])

##Fill missing values with MB AEGs as the count for MB and FP is almost similar 
LSH_Data['fingerprick alpha glycation event'].fillna(LSH_Data['MB alpha glycation events'], inplace=True)
LSH_Data['fingerprick beta glycation event'].fillna(LSH_Data['MB beta glycation events'], inplace=True)

##Fill missing values with '0' as the glycation value for MB is near to '0'
LSH_Data['fingerprick glycation HbB'] = LSH_Data['fingerprick glycation HbB'].fillna(0)
LSH_Data['fingerprick glycation HbA'] = LSH_Data['fingerprick glycation HbA'].fillna(0)

##Calculating and adding BMI to the DataFrame
##LSH_Data['BMI'] = 703 * (LSH_Data['Weight in pounds']/(LSH_Data['Height in inches']*LSH_Data['Height in inches']))

#Grouping FP beta glycation rate into groups
bins = [-1, 4, 5.6, 6.4, np.inf]
names = ['low', 'normal', 'pre-diabetic', 'diabetic']
d = dict(enumerate(names, 1))

LSH_Data['diabeticRange_fp_beta'] = np.vectorize(d.get)(np.digitize(LSH_Data['fingerprick glycation HbB'], bins))
LSH_Data['diabeticRange_fp_beta'] = pd.cut(LSH_Data['fingerprick glycation HbB'], bins, labels=names)

##Calculating Age from Birthday
LSH_Data['Age'] = LSH_Data['Birthday'].apply(lambda x: abs(pd.to_datetime(x).date() - date.today()).days/365)

##Dropping Birthday, Weight and Height as we calculated BMI, ID
LSH_Data = LSH_Data.drop(columns = ['ID#', 'Birthday'])

##Cleaning Exercise Frequency Column
LSH_Data['Exercise freq'][(LSH_Data['Exercise freq'] == 'occasionally')] = 'Occasionally'
LSH_Data['Exercise freq'][(LSH_Data['Exercise freq'] == 'rarely')] = 'Rarely'
LSH_Data['Exercise freq'][(LSH_Data['Exercise freq'] == 'regularly')] = 'Regularly'
LSH_Data['Exercise freq'][(LSH_Data['Exercise freq'] == 'never')] = 'Never'

##Cleaning Types of Fruit Veggies Column
LSH_Data['Diff types of fruit veggies in house'][(LSH_Data['Diff types of fruit veggies in house']=='04-Aug')] = '4 to 8'
LSH_Data['Diff types of fruit veggies in house'][(LSH_Data['Diff types of fruit veggies in house']=='0-3')] = '0 to 3'

##Cleaning Alcohol drinks per week column
LSH_Data['Alcohol drinks/week'][(LSH_Data['Alcohol drinks/week']=='04-Aug')] = '4 to 8'
LSH_Data['Alcohol drinks/week'][(LSH_Data['Alcohol drinks/week']=='Oct-17')] = '10-17'

##Cleaning Types of Exercise Type Column
LSH_Data['Exercise Type'][(LSH_Data['Exercise Type'] == 'walking')] = 'Walking'
LSH_Data['Exercise Type'][(LSH_Data['Exercise Type'] == 'Walking ')] = 'Walking'

LSH_Data['Exercise Type'][~LSH_Data['Exercise Type'].isin(['Walking','Cardio','Wt lifting',
                                                           'Running, biking, hiking, sports, weight lifting',
                                                           'Cardio', 'spinning', 'running/cardio/weightlifting',
                                                           'Home workout','Walk, run, weights',
                                                           'Various, walking, dance or aerobic, yoga, kayak'])] = 'other'

##Replacing Yes and No as 1 and 0
LSH_Data = LSH_Data.replace(('Yes', 'No'), (1, 0))

##Taking only features which are available without testing
LSH_Data_2 = LSH_Data[['Race', 'Ethnicity', 'Age', 'Weight in pounds', 'Height in inches', 'Birth Control', 
                       'Exercise freq', 'Exercise Type', 'Diff types of fruit veggies in house',
                      'Smoke', 'Alcohol drinks/week', 'Heart Disease', 'Hypothyrodism ', 'Asthma', 'Autoimmune',
                      'Depression', 'High Blood Pressure', 'High Cholesterol', 'Thyroid Disease',
                      'diabeticRange_fp_beta']]

##Dividing Categorical and Numerical Features seperately
#LSH_cat = LSH_Data[['Race', 'Ethnicity', 'Exercise freq', 'Exercise Type', 'Diff types of fruit veggies in house',
#                   'Alcohol drinks/week', 'diabeticRange_fp_beta']]
#LSH_num = LSH_Data.drop(['Race', 'Ethnicity', 'Exercise freq', 'Exercise Type', 'Diff types of fruit veggies in house',
#                         'Alcohol drinks/week', 'diabeticRange_fp_beta'], axis = 1)

exp_name = setup(data = LSH_Data_2,  target = 'diabeticRange_fp_beta')

best_model = compare_models()

#Create Logistic Regression Model
rf = create_model('rf')

#Hyperparameter Tuning
tuned_rf = tune_model(rf)

#Save the model
save_model(tuned_rf, model_name = 'deployment_aly6980')



