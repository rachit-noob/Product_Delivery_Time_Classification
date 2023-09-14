import streamlit as st
import pandas as pd
import joblib

# Load your model and dataset here
try:
    pipeline = joblib.load('SVM_pipeline_new.pkl')  # Replace with the correct pickle file path
    car = pd.read_csv('Encoded.csv')
except Exception as e:
    st.error(f"Error loading the model or dataset: {str(e)}")

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

# Map gender, warehouse_block, and mode_of_shipment to one-hot encoded values
gender_mapping = {'Male': [1, 0], 'Female': [0, 1]}
warehouse_mapping = {'A': [1, 0, 0, 0, 0], 'B': [0, 1, 0, 0, 0], 'C': [0, 0, 1, 0, 0], 'D': [0, 0, 0, 1, 0], 'F': [0, 0, 0, 0, 1]}
shipment_mapping = {'Flight': [1, 0, 0], 'Road': [0, 1, 0], 'Ship': [0, 0, 1]}

# Get the one-hot encoded values based on user selection
gender_encoded = gender_mapping.get(gender, [0, 0])
warehouse_encoded = warehouse_mapping.get(warehouse_block, [0, 0, 0, 0])
shipment_encoded = shipment_mapping.get(mode_of_shipment, [0, 0, 0])

# Predict when the user clicks the "Predict" button
if st.button('Predict'):
    input_data = pd.DataFrame({
        'Customer_care_calls': [customer_care_calls],
        'Customer_rating': [customer_rating],
        'Cost_of_the_Product': [cost_of_the_product],
        'Prior_purchases': [prior_purchases],
        'Product_importance': [product_importance],
        'Gender_Male': [gender_encoded[0]],
        'Gender_Female': [gender_encoded[1]],
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
    except Exception as e:
        # Handle any errors gracefully and display an error message
        st.error(f"Error making predictions: {str(e)}")
