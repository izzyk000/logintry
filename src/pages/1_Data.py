import streamlit as st
import pandas as pd

# Function to load data into session state if it doesn't exist
def load_data():
    if 'coffee_products_df' not in st.session_state:
        st.session_state.coffee_products_df = pd.read_excel('data/Coffee_Products.xlsx', engine='openpyxl')
    if 'customers_data_df' not in st.session_state:
        st.session_state.customers_data_df = pd.read_excel('data/Customers_Data.xlsx', engine='openpyxl')
    if 'sales_data_df' not in st.session_state:
        st.session_state.sales_data_df = pd.read_excel('data/Sales_Data.xlsx', engine='openpyxl')

# Call the load_data function to ensure data is loaded
load_data()

# Page header
st.header('Data Overview for Coffee Products Model')

# Section for Coffee Productspip
st.subheader('Coffee Products Data')
st.write("Below is the dataset used for coffee products in our model:")
st.dataframe(st.session_state.coffee_products_df)

# Section for Customer Data
st.subheader('Customer Data')
st.write("Below is the dataset used for customers data in our model:")
st.dataframe(st.session_state.customers_data_df)

# Section for Sales Data
st.subheader('Sales Data')
st.write("Below is the dataset used for sales data in our model:")
st.dataframe(st.session_state.sales_data_df)

# Optionally, you can add explanations or descriptions using markdown
st.markdown("""
This page displays the primary datasets used in the machine learning model for predicting coffee products performance.
- **Coffee Products Data**: Contains information about different coffee products.
- **Customer Data**: Includes details about customers.
- **Sales Data**: Records of sales transactions.
""")
