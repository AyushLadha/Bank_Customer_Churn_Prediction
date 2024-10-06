import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import streamlit as st

model = load_model('model.h5')

# loading encoders and scaler
with open('OneHotEncoder_Geography.pkl', 'rb') as file:
  OneHotEncoder_Geography = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
  label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
  scaler = pickle.load(file)

  # Stream lit
st.title('Bank Customer Churn Prediction')

# User input
geography = st.selectbox('Geography', OneHotEncoder_Geography.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', options=[
    ('0', 'No'), 
    ('1', 'Yes')
], format_func=lambda x: x[1])  # Using a lambda function to display 'No' or 'Yes'

# More descriptive options for active membership status
is_active_member = st.selectbox('Is Active Member', options=[
    ('0', 'No'),
    ('1', 'Yes')
], format_func=lambda x: x[1])  # Using a lambda function to display 'No' or 'Yes'

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Encoding geography 
encode_geo = OneHotEncoder_Geography.transform([[geography]]).toarray()
geo_encoder_df = pd.DataFrame(encode_geo, columns=OneHotEncoder_Geography.get_feature_names_out(['Geography']))

#combine one hot encoded data with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoder_df], axis=1)

# scaling input data
input_data_scale = scaler.transform(input_data)

# predicting churn
prediction = model.predict(input_data_scale)
prediction_proba = prediction[0][0]

st.write(f"Churn Probability: {prediction_proba:.2f}")

if prediction_proba > 0.5:
  st.write("Customer is likely to churn.")
else:
  st.write("Customer is not likely to churn.")

