# Databricks notebook source
# Import Dependecies
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

# COMMAND ----------

# Load data
url = "https://data.gov.sg/dataset/.../download"
df = pd.read_csv(url)

# Data preprocessing
data.dropna(inplace=True)  # Drop rows with missing values

# Structure the data for machine learning
X = data[['town', 'flat_type', 'floor_area_sqm', 'lease_commence_date']]
y = data['resale_price']


# COMMAND ----------

# Encode categorical variables
label_encoder = LabelEncoder()
X['town'] = label_encoder.fit_transform(X['town'])
X['flat_type'] = label_encoder.fit_transform(X['flat_type'])


# COMMAND ----------

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)


# COMMAND ----------

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)


# COMMAND ----------

# Function to predict resale price
def predict_price(town, flat_type, floor_area_sqm, lease_commence_date):
    # Preprocess input data if necessary
    input_data = [[town, flat_type, floor_area_sqm, lease_commence_date]]
    # Make prediction
    predicted_price = model.predict(input_data)
    return predicted_price[0]

# Streamlit UI
def main():
    st.title('Singapore Resale Flat Price Predictor')
    st.write('Enter details of the flat to predict resale price:')
    
    town = st.text_input('Town:')
    flat_type = st.selectbox('Flat Type:', ['1 Room', '2 Room', '3 Room', '4 Room', '5 Room', 'Executive', 'Multi-Generation'])
    floor_area_sqm = st.number_input('Floor Area (sqm):')
    lease_commence_date = st.number_input('Lease Commence Year:')
    
    if st.button('Predict Price'):
        predicted_price = predict_price(town, flat_type, floor_area_sqm, lease_commence_date)
        st.success(f'Predicted Resale Price: SGD {predicted_price:.2f}')

if __name__ == '__main__':
    main()

