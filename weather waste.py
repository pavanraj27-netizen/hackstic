# weather_waste_ai.py
# Run using:
# pip install streamlit pandas scikit-learn matplotlib
# streamlit run weather_waste_ai.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Waste Management AI", layout="wide")

st.title("🌦️ Smart Weather-Based Waste Management System")
st.write("AI-powered waste prediction and adaptive collection planning")

# ---------------------------------------------------
# CREATE SAMPLE DATASET
# ---------------------------------------------------

np.random.seed(42)

rows = 500

temperature = np.random.randint(20, 42, rows)
rainfall = np.random.randint(0, 120, rows)
humidity = np.random.randint(30, 100, rows)

# Synthetic waste generation formula
waste_generated = (
    temperature * 2
    + humidity * 1.5
    + rainfall * 0.8
    + np.random.randint(0, 50, rows)
)

data = pd.DataFrame({
    "Temperature": temperature,
    "Rainfall": rainfall,
    "Humidity": humidity,
    "WasteGenerated": waste_generated
})

# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------

X = data[["Temperature", "Rainfall", "Humidity"]]
y = data["WasteGenerated"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# ---------------------------------------------------
# SIDEBAR INPUT
# ---------------------------------------------------

st.sidebar.header("Enter Weather Conditions")

temp_input = st.sidebar.slider("Temperature (°C)", 20, 45, 35)
rain_input = st.sidebar.slider("Rainfall (mm)", 0, 150, 40)
humidity_input = st.sidebar.slider("Humidity (%)", 20, 100, 75)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

input_data = pd.DataFrame({
    "Temperature": [temp_input],
    "Rainfall": [rain_input],
    "Humidity": [humidity_input]
})

prediction = model.predict(input_data)[0]

# ---------------------------------------------------
# RISK DETECTION
# ---------------------------------------------------

risk = "Low"
action = "Normal waste collection schedule."

if prediction > 180:
    risk = "High"
    action = """
    • Deploy extra garbage trucks
    • Increase collection frequency
    • Prioritize organic waste zones
    """

elif prediction > 130:
    risk = "Medium"
    action = """
    • Monitor overflow-prone areas
    • Add temporary bins
    """

# ---------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Predicted Waste Generation")
    st.metric("Estimated Waste", f"{prediction:.2f} kg")

with col2:
    st.subheader("⚠️ Overflow Risk")
    st.metric("Risk Level", risk)

st.subheader("🚛 Recommended Actions")
st.write(action)

# ---------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------

st.subheader("📈 Waste vs Temperature")

fig, ax = plt.subplots(figsize=(8, 4))

ax.scatter(data["Temperature"], data["WasteGenerated"])
ax.set_xlabel("Temperature")
ax.set_ylabel("Waste Generated")
ax.set_title("Waste Generation Pattern")

st.pyplot(fig)

# ---------------------------------------------------
# RAW DATA
# ---------------------------------------------------

st.subheader("🗂️ Sample Dataset")
st.dataframe(data.head(20))

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.success("AI model successfully analyzed weather-based waste patterns.")
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)