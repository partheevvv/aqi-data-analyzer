# AQI Data Analyzer Web Application

A Streamlit web application that fetches, analyzes, and visualizes Air Quality Index (AQI) data for cities in India. The application allows users to select a state and city, then generates data visualizations and trains a machine learning model to predict AQI values.

![AQI Data Analyzer Screenshot](https://raw.githubusercontent.com/user/aqi_app/main/screenshot.png)

## Features

- **Data Retrieval**: Fetch AQI data from the Indian government's official API
- **Interactive UI**: Simple interface for selecting location and analysis parameters
- **Data Visualization**: 
  - Distribution of pollutant averages 
  - Correlation heatmaps
  - Min vs. Max pollutant scatter plots
- **Machine Learning**: 
  - Random Forest Regressor to predict average pollutant values
  - Model performance metrics and visualization
- **Data Export**: Download analyzed data as CSV files

## Installation

### Prerequisites
- Python 3.9, 3.10, or 3.11 recommended (Python 3.12 may require additional setup)
- pip (Python package manager)

### Setup Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/aqi_app.git
   cd aqi_app
   ```

2. Create a virtual environment:
   ```bash
   # For Windows
   python -m venv venv
   venv\Scripts\activate

   # For macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Note: If you encounter issues with Python 3.12, try:
   ```bash
   pip install setuptools
   pip install -r requirements.txt
   ```
   
   Or use the alternative requirements file:
   ```bash
   pip install -r requirements-loose.txt
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

5. Open your browser and navigate to the URL displayed in your terminal (typically http://localhost:8501)

## Usage

1. Select a state and city from the sidebar
2. Adjust the maximum records to fetch if needed
3. Click "Analyze AQI Data" to start the analysis
4. View the data overview, visualizations, and model results
5. Download the data as CSV if needed

## Example Locations

Some locations known to have data in the API:
- Delhi, Delhi
- Mumbai, Maharashtra
- Kolkata, West_Bengal
- Chennai, Tamil_Nadu
- Bangalore, Karnataka

## API Information

The application uses the official Indian government data portal API:
- Endpoint: https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69
- API Documentation: [Data.gov.in API Documentation](https://data.gov.in/apis)

## Acknowledgments

- Data provided by the Government of India's data portal
- Built with Streamlit, Pandas, Scikit-learn, and Seaborn
