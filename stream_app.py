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
warehouse_block = st.selectbox('Warehouse Block', ['A', 'B', 'C', 'D'])
mode_of_shipment = st.selectbox('Mode of Shipment', ['Flight', 'Road'])

# Map gender, warehouse_block, and mode_of_shipment to numeric values
gender_mapping = {'Male': 1, 'Female': 0}
warehouse_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
shipment_mapping = {'Flight': 1, 'Road': 0}

gender_numeric = gender_mapping.get(gender, 0)
warehouse_numeric = warehouse_mapping.get(warehouse_block, 0)
shipment_numeric = shipment_mapping.get(mode_of_shipment, 0)

# Predict when the user clicks the "Predict" button
if st.button('Predict'):
    input_data = pd.DataFrame({
        'Customer_care_calls': [customer_care_calls],
        'Customer_rating': [customer_rating],
        'Cost_of_the_Product': [cost_of_the_product],
        'Prior_purchases': [prior_purchases],
        'Product_importance': [product_importance],
        'Gender': [gender_numeric],
        'Discount_offered': [discount_offered],
        'Weight_in_gms': [weight_in_gms],
        'Warehouse_block_A': [1 if warehouse_numeric == 0 else 0],
        'Warehouse_block_B': [1 if warehouse_numeric == 1 else 0],
        'Warehouse_block_C': [1 if warehouse_numeric == 2 else 0],
        'Warehouse_block_D': [1 if warehouse_numeric == 3 else 0],
        'Mode_of_Shipment_Flight': [shipment_numeric],
        'Mode_of_Shipment_Road': [1 if shipment_numeric == 0 else 0]
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
