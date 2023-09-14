import streamlit as st
import pandas as pd
import joblib

# Load your model and dataset here
try:
    pipeline = joblib.load('SVM_pipeline_new.pkl')  # Replace with the correct pickle file path
except FileNotFoundError:
    st.error("Model file not found. Please upload the model file.")
except Exception as e:
    st.error(f"Error loading the model: {str(e)}")

# Define a Streamlit app
st.title('Machine Learning Prediction')

# Define input fields for prediction
customer_care_calls = st.number_input('Customer Care Calls', min_value=0)
customer_rating = st.number_input('Customer Rating', min_value=1, max_value=5)
cost_of_the_product = st.number_input('Cost of the Product', step=0.01)
prior_purchases = st.number_input('Prior Purchases', min_value=0)
product_importance = st.number_input('Product Importance', min_value=0)
gender = st.selectbox('Gender', ['Male', 'Female'])
discount_offered = st.number_input('Discount Offered', step=0.01)
weight_in_gms = st.number_input('Weight in grams', step=0.01)
warehouse_block = st.selectbox('Warehouse Block', ['A', 'B', 'C', 'D', 'F'])
mode_of_shipment = st.selectbox('Mode of Shipment', ['Flight', 'Road', 'Ship'])

# Get one-hot encoded values based on user selection
gender_encoded = [1 if gender == 'Male' else 0]
warehouse_encoded = [0] * 5
warehouse_encoded['ABCDE'.index(warehouse_block)] = 1
shipment_encoded = [1 if mode_of_shipment == mode else 0 for mode in ['Flight', 'Road', 'Ship']]

# Predict when the user clicks the "Predict" button
if st.button('Predict'):
    if not all([customer_care_calls, customer_rating, cost_of_the_product, prior_purchases, product_importance,
                discount_offered, weight_in_gms]):
        st.warning("Please fill in all required fields.")
    else:
        input_data = pd.DataFrame({
            'Customer_care_calls': [customer_care_calls],
            'Customer_rating': [customer_rating],
            'Cost_of_the_Product': [cost_of_the_product],
            'Prior_purchases': [prior_purchases],
            'Product_importance': [product_importance],
            'Gender': gender_encoded,
            'Discount_offered': [discount_offered],
            'Weight_in_gms': [weight_in_gms],
            'Warehouse_block_A': [warehouse_encoded[0]],
            'Warehouse_block_B': [warehouse_encoded[1]],
            'Warehouse_block_C': [warehouse_encoded[2]],
            'Warehouse_block_D': [warehouse_encoded[3]],
            'Warehouse_block_F': [warehouse_encoded[4]],
            'Mode_of_Shipment_Flight': [shipment_encoded[0]],
            'Mode_of_Shipment_Road': [shipment_encoded[1]],
            'Mode_of_Shipment_Ship': [shipment_encoded[2]]
        })

        try:
            # Use the loaded model to make predictions
            predictions = pipeline.predict(input_data)

            if predictions[0] == 1:
                st.write('Product Reached With Delay')
            else:
                st.write('Product Reached on Time')
        except ValueError as e:
            # Handle prediction-related errors gracefully and display an error message
            st.error(f"Error making predictions: {str(e)}")
