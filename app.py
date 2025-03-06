import streamlit as st
import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Set page configuration
st.set_page_config(page_title="AQI Data Analyzer", layout="wide")

# API Endpoint
BASE_URL = "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
API_KEY = "579b464db66ec23bdd000001586fe5bfeda64e5c55d27328dcc242a8"git a

def fetch_aqi_data(country="India", state="Delhi", city="Delhi", limit=10000):
    """
    Fetches AQI data from the API by state and city.
    Uses official API parameters for precise filtering
    """
    params = {
        'api-key': API_KEY,
        'format': 'json',
        'limit': limit,
        'filters[country]': country,
        'filters[state]': state,
        'filters[city]': city  # Add city filter parameter
    }

    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    return response.json()

def parse_aqi_data(raw_data):
    """Parses the JSON data into a pandas DataFrame and cleans it."""
    if 'records' not in raw_data or not raw_data['records']:
        raise ValueError("No records found in the API response. Please check your filter parameters.")

    df = pd.DataFrame(raw_data['records'])

    # Convert numerical fields relevant to AQI analysis
    numeric_cols = ['min_value', 'max_value', 'avg_value']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows where target variable 'avg_value' is NaN
    df.dropna(subset=['avg_value'], inplace=True)

    # Fill missing values in other numeric columns with median
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    return df

def perform_eda(df):
    """Performs exploratory data analysis with visualizations."""
    
    # Create description and display dataframe
    st.subheader("Data Overview")
    st.dataframe(df.head())
    
    # Display descriptive statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe())
    
    # Create visualizations
    st.subheader("Data Visualizations")
    
    # Create columns for the charts
    col1, col2 = st.columns(2)
    
    # Histogram of pollutant averages
    with col1:
        st.write("Distribution of Pollutant Average")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(df['avg_value'], kde=True, bins=20, ax=ax)
        ax.set_title("Distribution of Pollutant Average")
        ax.set_xlabel("Pollutant Average Value")
        st.pyplot(fig)
    
    # Correlation Heatmap
    with col2:
        st.write("Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        corr = df[['min_value', 'max_value', 'avg_value']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title("Feature Correlation Heatmap")
        st.pyplot(fig)
    
    # Scatter Plot
    st.write("Min vs. Max Pollutant Values")
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'pollutant_id' in df.columns:
        sns.scatterplot(data=df, x='min_value', y='max_value', hue='pollutant_id', ax=ax)
    else:
        sns.scatterplot(data=df, x='min_value', y='max_value', ax=ax)
    ax.set_title("Min vs. Max Pollutant Values")
    ax.set_xlabel("Min Pollutant Value")
    ax.set_ylabel("Max Pollutant Value")
    st.pyplot(fig)

def train_model(df):
    """
    Trains a Random Forest Regressor to predict AQI (avg_value).
    Using only min_value and max_value as features (latitude, longitude removed).
    """
    st.subheader("Model Training Results")
    
    # Define target and features
    target = 'avg_value'
    X = df[['min_value', 'max_value']]
    y = df[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)

    # Random Forest Model
    with st.spinner("Training model... This may take a moment."):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Evaluation metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Display metrics
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            st.metric("Train MSE", f"{train_mse:.2f}")
            st.metric("Test MSE", f"{test_mse:.2f}")
        
        with metrics_col2:
            st.metric("Train R²", f"{train_r2:.2f}")
            st.metric("Test R²", f"{test_r2:.2f}")

    return model, X_test, y_test

def visualize_results(model, X_test, y_test):
    """Generates residual plots and actual vs. predicted plots."""
    st.subheader("Model Visualization")
    
    y_pred = model.predict(X_test)
    
    col1, col2 = st.columns(2)
    
    # Residual Plot
    with col1:
        st.write("Residual Plot")
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=y_pred, y=residuals, ax=ax)
        ax.axhline(y=0, color='red', linestyle='--')
        ax.set_title("Residual Plot")
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals")
        st.pyplot(fig)

    # Actual vs. Predicted
    with col2:
        st.write("Actual vs Predicted Values")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        # Diagonal line for reference
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Actual vs Predicted Pollutant Values")
        st.pyplot(fig)

def main():
    # App title and description
    st.title("Air Quality Index (AQI) Data Analyzer")
    st.markdown("""
    This application allows you to fetch and analyze Air Quality Index (AQI) data for different cities in India.
    Enter a state and city to retrieve the data and generate visualizations and predictive models.
    """)
    
    # Sidebar inputs
    st.sidebar.header("Input Parameters")
    
    # State input (with default)
    state = st.sidebar.text_input("State", "Delhi")
    
    # City input (with default)
    city = st.sidebar.text_input("City", "Delhi")
    
    # Limit for API records
    limit = st.sidebar.slider("Max Records to Fetch", 1000, 10000, 5000, 1000)
    
    # Submit button
    submit_button = st.sidebar.button("Analyze AQI Data")
    
    if submit_button:
        try:
            # Show loading message
            with st.spinner(f"Fetching AQI data for {city}, {state}..."):
                # 1. Fetch data
                raw_data = fetch_aqi_data(
                    country="India",
                    state=state,
                    city=city,
                    limit=limit
                )
            
            # 2. Parse the city-specific data
            try:
                df = parse_aqi_data(raw_data)
                st.success(f"Successfully retrieved {len(df)} records for {city}, {state}")
                
                # 3. Perform EDA
                perform_eda(df)
                
                # 4. Train model
                model, X_test, y_test = train_model(df)
                
                # 5. Visualize results
                visualize_results(model, X_test, y_test)
                
                # 6. Provide download option for the data
                st.subheader("Download Data")
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="{city}_{state}_aqi_data.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
                
            except ValueError as e:
                st.error(f"Error: {e}")
                st.info("Try a different state or city name, or check API availability.")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please check your internet connection and try again.")
    
    # Show a message if the app is just loaded without submission
    else:
        st.info("Enter state and city names, then click 'Analyze AQI Data' to begin analysis.")
        
        # Provide some example cities that work with the API
        st.markdown("""
        ### Example Locations
        Try these locations that are known to have data in the API:
        - Delhi, Delhi
        - Mumbai, Maharashtra
        - Kolkata, West Bengal
        - Chennai, Tamil Nadu
        - Bangalore, Karnataka
        """)

if __name__ == "__main__":
    main()