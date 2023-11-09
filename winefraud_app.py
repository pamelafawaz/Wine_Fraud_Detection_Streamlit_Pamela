import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained SVM model
with open('wine_fraud_svm_model_pam.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the scaler
with open('wine_scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Function to preprocess the data
def preprocess_input(data):
    scaled_data = scaler.transform(data)
    return scaled_data

# Streamlit app
def main():
    st.title("Wine Fraud Detection")
    


    # Create inputs for all the features
    fixed_acidity = st.number_input('Fixed Acidity', min_value=0.0, max_value=20.0, value=7.4, step=0.1)
    volatile_acidity = st.number_input('Volatile Acidity', min_value=0.0, max_value=2.0, value=0.7, step=0.01)
    citric_acid = st.number_input('Citric Acid', min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    residual_sugar = st.number_input('Residual Sugar', min_value=0.0, max_value=100.0, value=2.0, step=0.1)
    chlorides = st.number_input('Chlorides', min_value=0.0, max_value=1.0, value=0.08, step=0.001)
    free_sulfur_dioxide = st.number_input('Free Sulfur Dioxide', min_value=0, max_value=300, value=30, step=1)
    total_sulfur_dioxide = st.number_input('Total Sulfur Dioxide', min_value=0, max_value=500, value=150, step=1)
    density = st.number_input('Density', min_value=0.0, max_value=2.0, value=0.99, step=0.0001)
    pH = st.number_input('pH', min_value=0.0, max_value=14.0, value=3.0, step=0.01)
    sulphates = st.number_input('Sulphates', min_value=0.0, max_value=2.0, value=0.5, step=0.01)
    alcohol = st.number_input('Alcohol', min_value=0.0, max_value=100.0, value=10.0, step=0.1)
    wine_type = st.selectbox('Type', ['red', 'white'])
    type_white = 1 if wine_type == 'white' else 0

    input_data = pd.DataFrame([[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
        free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol, wine_type
    ]], columns=[
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
        'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'type'
    ])

    input_data = pd.get_dummies(input_data, columns=['type'], drop_first=True)

    if 'type_white' not in input_data.columns:
        input_data['type_white'] = 0 if wine_type == 'red' else 1

    if st.button('Predict'):
        preprocessed = preprocess_input(input_data)
        prediction = model.predict(preprocessed)
        
        if prediction[0] == 1:
            st.subheader("The wine is likely to be Fraudulent.")
        else:
            st.subheader("The wine is likely to be Legitimate.")

if __name__ == '__main__':
    main()


