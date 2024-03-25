import streamlit as st
from joblib import load
import pandas as pd
import plotly.express as px
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

# Load the YAML configuration
with open('src/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Initialize the authentication system
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'], 
    config['cookie']['key'], 
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# Attempt to log the user in and capture the return values
try:
    name, authentication_status, username = authenticator.login()
except Exception as e:
    st.error(f"Login Error: {e}")

# Amend the logout function to check for the cookie before deletion
def safe_logout(authenticator):
    try:
        # Check if the cookie exists before attempting to delete
        if authenticator.cookie_manager.cookies.get(authenticator.cookie_name):
            authenticator.logout("Logout", "sidebar")
    except Exception as e:
        st.error(f"Logout Error: {e}")

# Attempt to log the user in and capture the return values
# ... (your existing code for login attempt)

# Handling different authentication states
if authentication_status:


    # Place the rest of your application logic here, as shown in your original code block
    # This includes loading models, displaying UI elements, handling user inputs, etc.
    

    # Load your model and encoding dictionaries
    model = load('models/your_model.joblib')
    origin_means = load('models/origin_means.joblib')
    roastlevel_means = load('models/roastlevel_means.joblib')
    feature_importance_data = load('models/feature_importance.joblib')  # Loading the feature importance data

    # Title and introduction
    # User is authenticated; proceed with the application logic...
    st.write(f'Hi *{name}*')
    st.title('Welcome to the Saigon Bean Bazaar Sales Estimator!')
    st.markdown("""

    Discover the sales potential of new coffee beans with our AI-driven estimator. Leveraging historical sales data from Saigon Bean Bazaar, this tool provides insights into how well new bean varieties might perform in our store. 

    Simply input details about the coffee beans you're considering, and let our model predict their sales success. Whether you're evaluating new origins, roast levels, or pricing strategies, our estimator is here to guide your decisions with data-driven confidence.

    **Get started** by entering the coffee bean characteristics below and press **Predict** to see the magic happen!
    """)


    # Sidebar - About description
    safe_logout(authenticator)
    st.sidebar.header('About Saigon Bean Bazaar Estimator')
    st.sidebar.info("""
    This predictive tool is designed exclusively for Saigon Bean Bazaar, a leading e-commerce store specializing in premium coffee beans. By analyzing extensive past sales data, our model assesses factors like bean origin, roast level, and pricing to forecast the market appeal of upcoming coffee beans.
    """)
    


    # User inputs
    origin_input = st.selectbox('Select Origin', options=list(origin_means.keys()), 
                                help='Choose the origin of the coffee bean.')
    roast_level_input = st.selectbox('Select Roast Level', options=list(roastlevel_means.keys()), 
                                    help='Choose the roast level.')
    price = st.slider('Price ($)', min_value=5.0, max_value=60.0, value=30.0, step=0.5,
                    help='Set the price of the coffee bean.')
        
    # Prediction and display results directly
    if st.button('Predict', key='predict_button'):
        origin_encoded = origin_means[origin_input]
        roast_level_encoded = roastlevel_means[roast_level_input]
        
        # Create a DataFrame with the correct feature names and the input data
        input_data = pd.DataFrame({
            'Origin_TargetEncoded': [origin_encoded],
            'RoastLevel_TargetEncoded': [roast_level_encoded],
            'Price': [price]
        })
        
        # Use the DataFrame for prediction
        raw_prediction = model.predict(input_data)
        
        # Function to adjust the prediction based on price outliers
        def adjust_prediction_for_price_outliers(prediction, price, high_price_threshold=40, low_price_threshold=10):
            if price > high_price_threshold:
                return 'Low'
            elif price < low_price_threshold:
                return 'High'
            else:
                return prediction
        
        # Adjust the prediction based on the price outliers
        adjusted_prediction = adjust_prediction_for_price_outliers(raw_prediction[0], price)

        # Display the result using markdown
        if adjusted_prediction == 'High':
            prediction_text = '### Prediction: High\n\nThis indicates a strong sales potential for the selected coffee bean characteristics.'
        elif adjusted_prediction == 'Medium':
            prediction_text = '### Prediction: Medium\n\nThis indicates a moderate sales potential. Consider adjusting price or targeting specific markets.'
        else:  # Low
            prediction_text = '### Prediction: Low\n\nThis suggests the sales potential is below average. Review the characteristics and pricing strategy.'

        st.markdown(prediction_text)

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')

elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')
