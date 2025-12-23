import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 1. Page Config ---
st.set_page_config(page_title="Linear Regression Dashboard", layout="centered")

# --- 2. Load CSS ---
def load_css(file_name):
    try:
        with open(file_name, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("style.css not found. Proceeding with default styles.")

load_css("style.css")

# --- 3. Header ---
st.markdown(""" 
<div class="card">
    <h1>Linear Regression Model</h1>
    <p style="color: #333;">Predicting <b>Tip Amount</b> based on <b>Total Bill</b> using the Seaborn Tips Dataset</p>
</div>
""", unsafe_allow_html=True)

# --- 4. Data Operations ---
@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()

# Dataset Preview Section
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

# --- 5. Model Logic ---
# Prepare features and target
X, y = df[["total_bill"]], df["tip"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Metrics Calculation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Adjusted R2 Calculation
n = len(y_test)
p = X_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# --- 6. Performance Metrics UI ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance")

c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae: .2f}")
c2.metric("RMSE", f"{rmse: .2f}")

c3, c4 = st.columns(2)
c3.metric("R2", f"{r2: .3f}")
c4.metric("adj R2", f"{adj_r2: .3f}")
st.markdown('</div>', unsafe_allow_html=True)

# --- 7. Visualization ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Regression Analysis: Total Bill vs Tip")

fig, ax = plt.subplots(figsize=(10, 6))
# Scatter plot of actual data
ax.scatter(df["total_bill"], df["tip"], alpha=0.5, color="#3498db", label="Actual Data")

# Regression Line
# Create a range of values for a smooth line
x_range = np.linspace(df["total_bill"].min(), df["total_bill"].max(), 100).reshape(-1, 1)
x_range_scaled = scaler.transform(x_range)
y_range_pred = model.predict(x_range_scaled)

ax.plot(x_range, y_range_pred, color='#e74c3c', linewidth=3, label="Regression Line")

ax.set_xlabel("Total Bill ($)")
ax.set_ylabel("Tip Amount ($)")
ax.legend()
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# --- 8. Model Intercept & Co-efficient (From Image) ---
# # m & c
st.markdown(f"""
<div class="card">
<h3> Model Intercept & Co-efficient</h3>
<p> <b> Co-efficient: </b> {model.coef_[0]:.3f}<br>
<b> Intercept: </b> {model.intercept_:.3f}</p>
</div>
""", unsafe_allow_html = True)

# --- 9. Prediction Section (From Image) ---
# # Prediction
st.markdown('<div class class = "card">', unsafe_allow_html = True)
st.subheader("Predict Tip Amount")

bill = st.slider("Total Bill $", float(df.total_bill.min()), float(df.total_bill.max()), 30.0)
tip = model.predict(scaler.transform([[bill]]))[0]

st.markdown(f'<div class = "prediction-box"> Predict Tip: $ {tip:.2f} </div>', unsafe_allow_html = True)

st.markdown('</div>', unsafe_allow_html = True)