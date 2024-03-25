

import pandas as pd
import numpy as np
import os
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


print("Current Working Directory:", os.getcwd())

# Load data
coffee_products_df = pd.read_excel('data/Coffee_Products.xlsx', engine='openpyxl')
customers_data_df = pd.read_excel('data/Customers_Data.xlsx', engine='openpyxl')
sales_data_df = pd.read_excel('data/Sales_Data.xlsx', engine='openpyxl')




import pandas as pd
import numpy as np
import os
import streamlit as st  # Import Streamlit
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

def load_data():
    # Check if running in Streamlit environment
    if 'st' in globals():
        # Use Streamlit session state to cache the data
        if 'coffee_products_df' not in st.session_state:
            st.session_state.coffee_products_df = pd.read_excel('data/Coffee_Products.xlsx', engine='openpyxl')
        if 'customers_data_df' not in st.session_state:
            st.session_state.customers_data_df = pd.read_excel('data/Customers_Data.xlsx', engine='openpyxl')
        if 'sales_data_df' not in st.session_state:
            st.session_state.sales_data_df = pd.read_excel('data/Sales_Data.xlsx', engine='openpyxl')
        # Return data from session state
        return st.session_state.coffee_products_df, st.session_state.customers_data_df, st.session_state.sales_data_df
    else:
        # Directly load data if not in Streamlit environment
        return (
            pd.read_excel('data/Coffee_Products.xlsx', engine='openpyxl'),
            pd.read_excel('data/Customers_Data.xlsx', engine='openpyxl'),
            pd.read_excel('data/Sales_Data.xlsx', engine='openpyxl')
        )

# Use the load_data function
coffee_products_df, customers_data_df, sales_data_df = load_data()


#Calculate monthly sales per product as part of Y variable

# Preparing the dataset
total_sales_per_product = sales_data_df.groupby('ProductID')['Quantity'].sum()
# Ensure 'LaunchDate' is a datetime column
coffee_products_df['LaunchDate'] = pd.to_datetime(coffee_products_df['LaunchDate'])
# Calculate the difference in months
end_date = pd.Timestamp('2023-12-31')
coffee_products_df['MonthsOnMarket'] = ((end_date.year - coffee_products_df['LaunchDate'].dt.year) * 12 + end_date.month - coffee_products_df['LaunchDate'].dt.month)

average_monthly_sales = pd.merge(total_sales_per_product.reset_index(), coffee_products_df[['ProductID', 'MonthsOnMarket']], on='ProductID')
average_monthly_sales['AvgMonthlySales'] = average_monthly_sales['Quantity'] / average_monthly_sales['MonthsOnMarket']


# Calculate % of the product quantity sold vs all products sold as part of Y variable

# Step 1: Calculate Total Sales for Each Month
sales_data_df['YearMonth'] = sales_data_df['SaleDate'].dt.to_period('M')  # Ensure YearMonth column is present
monthly_total_sales = sales_data_df.groupby('YearMonth')['Quantity'].sum().reset_index(name='TotalMonthlySales')

# Merge the monthly total sales back to the original sales data for calculation of shares
sales_data_with_monthly_totals = pd.merge(sales_data_df, monthly_total_sales, on='YearMonth')

# Step 2: Calculate Each Product's Monthly Sales Share
sales_data_with_monthly_totals['SalesShare'] = sales_data_with_monthly_totals['Quantity'] / sales_data_with_monthly_totals['TotalMonthlySales']

# Note: At this point, each sale has a 'SalesShare' representing its proportion of the total monthly sales.


# For analysis, you might want to aggregate this information back to the product level, 
# calculating an average sales share for each product across all months it was sold.
product_sales_share = sales_data_with_monthly_totals.groupby('ProductID')['SalesShare'].mean().reset_index()

# Merge this sales share information back to your average_monthly_sales DataFrame or any other DataFrame
# you're planning to use for further analysis or categorization.
average_monthly_sales_with_share = pd.merge(average_monthly_sales, product_sales_share, on='ProductID')
# Calculate the composite metric by multiplying 'AvgMonthlySales' by 'SalesShare'
average_monthly_sales_with_share['CompositeMetric'] = average_monthly_sales_with_share['AvgMonthlySales'] * average_monthly_sales_with_share['SalesShare']

# Define thresholds for categorization based on quantiles
low_threshold, high_threshold = average_monthly_sales_with_share['CompositeMetric'].quantile([0.33, 0.66])

# Function to categorize products
def categorize_performance(metric):
    if metric <= low_threshold:
        return 'Low'
    elif metric <= high_threshold:
        return 'Medium'
    else:
        return 'High'

# Apply categorization
average_monthly_sales_with_share['PerformanceCategory'] = average_monthly_sales_with_share['CompositeMetric'].apply(categorize_performance)

# Merge composite_df with coffee_products_df to include product attributes
final_df = pd.merge(average_monthly_sales_with_share, coffee_products_df, on='ProductID')


# Target encoding
final_df['Origin_TargetEncoded'] = final_df['Origin'].map(final_df.groupby('Origin')['CompositeMetric'].mean().to_dict())
final_df['RoastLevel_TargetEncoded'] = final_df['RoastLevel'].map(final_df.groupby('RoastLevel')['CompositeMetric'].mean().to_dict())

# Feature and target variable preparation
features = final_df[['Origin_TargetEncoded', 'RoastLevel_TargetEncoded', 'Price']]
target = final_df['PerformanceCategory']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=42)

# Model training
model = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42))

# Model evaluation (optional for the training script)
# Note: Adjust parameter grid as needed
param_grid = {
    'randomforestclassifier__n_estimators': [10, 50, 100, 200],
    'randomforestclassifier__max_depth': [None, 10, 20, 30]
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_


print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


# Assuming this is the result from your model training process
feature_importance = best_model.named_steps['randomforestclassifier'].feature_importances_
features = ['Origin_TargetEncoded', 'RoastLevel_TargetEncoded', 'Price']  # Your feature names

# Combine the features names with their importance scores
feature_importance_data = dict(zip(features, feature_importance))

# Save the feature importance data
dump(feature_importance_data, 'models/feature_importance.joblib')
    

y_pred_best = best_model.predict(X_test)
print(classification_report(y_test, y_pred_best))

# Ensure the models directory exists
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)

# Save the model and encoding dictionaries
dump(best_model, 'models/your_model.joblib')
dump(final_df.groupby('Origin')['CompositeMetric'].mean().to_dict(), 'models/origin_means.joblib')
dump(final_df.groupby('RoastLevel')['CompositeMetric'].mean().to_dict(), 'models/roastlevel_means.joblib')
