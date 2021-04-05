# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 15:47:56 2021

@author: praka
"""

from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('deployment_aly6980')

def predict(model, input_df):
    predictions_df = predict_model(estimator = model, data = input_df)
    predictions = predictions_df['Label'][0]
    return predictions


from PIL import Image
company_logo = Image.open('company_logo.jfif')
college_logo = Image.open('NEU_Banner.jpg')

st.image(company_logo, use_column_width=False)

add_selectbox = st.sidebar.selectbox("How would you like to predict?", ("Online", "Batch"))

st.sidebar.info("This app is created to predict if a female is Normal, pre-diabetic or diabetic patient")

st.sidebar.success("Instructed by Dr. Ghazal Tariri")

st.sidebar.image(college_logo)

st.title("Diabetes Prediction using Non-Test data")

if add_selectbox == 'Online':
    
    Race = st.selectbox('Race', ['White', 'Black or African American', 'American Indian or Alaskan Native', 'Native Hawaiian or Other Pacific Islander', 'Other'])
    
    Ethnicity = st.selectbox('Ethnicity', ['Not Hispanic or Latino', 'Hispanic or Latino'])
    
    Age = st.number_input('Age', min_value = 1, max_value = 100, value = 30)
    
    Height = st.number_input('Height in Inches', min_value = 30, max_value = 100, value = 65)
    
    Weight = st.number_input('Weight in Pounds', min_value = 25, max_value = 500, value = 150)
    
    Exercise_freq = st.selectbox('Exercise Frequency', ['Occasionally', 'Regularly', 'Rarely', 'Never'])
    
    Exercise_type = st.selectbox('Exercise Type', ['other', 'Walking', 'Running, biking, hiking, sports, weight lifting', 
                                                   'Cardio', 'spinning', 'running/cardio/weightlifting', 'Walk, run, weights', 'Various, walking, dance or aerobic, yoga, kayak'])
    
    #Diff_types_of_fruit_veggies_in_house = st.selectbox('Different Types of Fruits & Veggies in house',
    #                                                    ['4 to 8', '0 to 3', '8', '10', '6', '10+', '4', '20', '9 or more', '7', 'Lots'])
    
    Alcohol_drinks_per_week = st.selectbox('Alcohol Drinks per Week', ['0-3', '0', '4 to 8', '1', '0-2', '6', '5', '2', '8', '9 or more', '10-17', '10'])
    
    if st.checkbox('Smoker'):
        smoker = 1
    else:
        smoker = 0
        
    if st.checkbox('Birth Control'):
        Birth_Control = 1
    else:
        Birth_Control = 0
        
    if st.checkbox('Heart Disease'):
        heart_disease = 1
    else:
        heart_disease = 0

    if st.checkbox('Hypothyrodism'):
        Hypothyrodism = 1
    else:
        Hypothyrodism = 0

    if st.checkbox('Asthma'):
        Asthma = 1
    else:
        Asthma = 0

    if st.checkbox('Autoimmune'):
        Autoimmune = 1
    else:
        Autoimmune = 0

    if st.checkbox('Depression'):
        Depression = 1
    else:
        Depression = 0
        
    if st.checkbox('High Blood Pressure'):
        High_Blood_Pressure = 1
    else:
        High_Blood_Pressure = 0
        
    if st.checkbox('High Cholesterol'):
        High_Cholesterol = 1
    else:
        High_Cholesterol = 0
        
    if st.checkbox('Thyroid Disease'):
        Thyroid_Disease = 1
    else:
        Thyroid_Disease = 0
        
    
    output = ""
    
    
    input_dict = {'Race': Race, 'Ethnicity': Ethnicity, 'Age': Age, 'Weight in pounds': Weight, 'Height in inches': Height, 
                  'Birth Control': Birth_Control, 'Exercise freq': Exercise_freq, #'Diff types of fruit veggies in house': Diff_types_of_fruit_veggies_in_house, 
                  'Exercise Type': Exercise_type, 'Smoke': smoker, 'Alcohol drinks/week': Alcohol_drinks_per_week, 'Heart Disease': heart_disease, 'Hypothyrodism ': Hypothyrodism, 
                  'Asthma': Asthma, 'Autoimmune': Autoimmune, 'Depression': Depression, 'High Blood Pressure': High_Blood_Pressure, 'High Cholesterol': High_Cholesterol, 
                  'Thyroid Disease': Thyroid_Disease}
    
    input_df = pd.DataFrame([input_dict])
    
    if st.button("Predict"):
        output = predict(model=model, input_df = input_df)
        output = str(output)
        
    st.success('The diabetic condition of the patient is {}'.format(output))

if add_selectbox == 'Batch':
    
    file_upload = st.file_uploader("Upload csv file for predictions", type = ["csv"])
    
    if file_upload is not None:
        data = pd.read_csv(file_upload)
        predictions = predict_model(estimator = model, data = data)
        st.write(predictions)
            
        
    
    
    