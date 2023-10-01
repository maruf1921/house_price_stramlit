import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("housing.csv")

# Drop rows with missing values
data.dropna(inplace=True)

# One-hot encode the 'ocean_proximity' column
data_encoded = data.join(pd.get_dummies(data.ocean_proximity)).drop(['ocean_proximity'], axis=1)

# Rename the '<1H OCEAN' column to '<One_hour OCEAN'
data_encoded.rename(columns={'<1H OCEAN': '<One_hour OCEAN'}, inplace=True)

# Apply logarithm transformations
data_encoded['total_rooms'] = np.log(data_encoded['total_rooms'] + 1)
data_encoded['total_bedrooms'] = np.log(data_encoded['total_bedrooms'] + 1)
data_encoded['households'] = np.log(data_encoded['households'] + 1)
data_encoded['population'] = np.log(data_encoded['population'] + 1)

# Create Streamlit web app
st.title("California Housing Data Exploration")

# Display dataset information
st.subheader("Dataset Information")
st.write(data_encoded.describe())
st.write(data_encoded.head())

# Display histograms
st.subheader("Histograms")
fig, ax = plt.subplots(figsize=(15, 8))
data_encoded.hist(ax=ax)
st.pyplot(fig)

# Display scatterplot
st.subheader("Scatterplot: Latitude vs. Longitude")
fig, ax = plt.subplots(figsize=(15, 8))
sns.scatterplot(x="latitude", y="longitude", data=data_encoded, hue="median_house_value", palette="coolwarm", ax=ax)
st.pyplot(fig)

# Display final correlation heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(15, 8))
sns.heatmap(data_encoded.corr(), annot=True, cmap='YlGnBu', ax=ax)
st.pyplot(fig)

# Create a linear regression model
L_reg = LinearRegression()

X = data_encoded.drop(['median_house_value'], axis=1)
y = data_encoded['median_house_value']

# Train the model
L_reg.fit(X, y)

# Create a form to input features for prediction in the sidebar
st.sidebar.subheader("Predict Median House Value")

# User inputs for numerical features
longitude = st.sidebar.number_input("Longitude (-124.35 to -114.31)", min_value=-124.35, max_value=-114.31)
latitude = st.sidebar.number_input("Latitude (32.54 to 41.95)", min_value=32.54, max_value=41.95)
housing_median_age = st.sidebar.number_input("Housing Median Age (1 to 52)", min_value=1, max_value=52)
total_rooms = st.sidebar.number_input("Total Rooms (1.0986 to 10.5795)", min_value=1.0986, max_value=10.5795)
total_bedrooms = st.sidebar.number_input("Total Bedrooms (0.6931 to 8.7712)", min_value=0.6931, max_value=8.7712)
population = st.sidebar.number_input("Population (1.3863 to 10.4824)", min_value=1.3863, max_value=10.4824)
households = st.sidebar.number_input("Households (0.6931 to 8.7133)", min_value=0.6931, max_value=8.7133)
median_income = st.sidebar.number_input("Median Income (0.4999 to 15.0001)", min_value=0.4999, max_value=15.0001)


# User inputs for categorical features
ocean_proximity = st.sidebar.selectbox("Select Ocean Proximity", ["<One_hour OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"])

# Map user input to 1 or 0 based on selected category
INLAND = 1 if ocean_proximity == "INLAND" else 0
ISLAND = 1 if ocean_proximity == "ISLAND" else 0
NEAR_BAY = 1 if ocean_proximity == "NEAR BAY" else 0
NEAR_OCEAN = 1 if ocean_proximity == "NEAR OCEAN" else 0

# Create input data dictionary
input_data = {
    'longitude': [longitude],
    'latitude': [latitude],
    'housing_median_age': [housing_median_age],
    'total_rooms': [total_rooms],
    'total_bedrooms': [total_bedrooms],
    'population': [population],
    'households': [households],
    'median_income': [median_income],
    '<One_hour OCEAN': 0,  # Use the mapped values
    'INLAND': [INLAND],
    'ISLAND': [ISLAND],
    'NEAR BAY': [NEAR_BAY],
    'NEAR OCEAN': [NEAR_OCEAN]
}

# Create a DataFrame from the input data
input_df = pd.DataFrame(input_data)

# Preprocess the input data
input_df['total_rooms'] = np.log(input_df['total_rooms'] + 1)
input_df['total_bedrooms'] = np.log(input_df['total_bedrooms'] + 1)
input_df['households'] = np.log(input_df['households'] + 1)
input_df['population'] = np.log(input_df['population'] + 1)

# Make the prediction
predicted_value = L_reg.predict(input_df)

# Display the predicted median house value in the main content area
st.subheader("Predicted Median House Value")
st.write(f"Predicted Value: ${predicted_value[0]:,.2f}")
