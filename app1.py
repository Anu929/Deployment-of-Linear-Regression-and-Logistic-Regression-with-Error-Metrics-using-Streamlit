import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
st.title("📊 Regression Analysis using Streamlit")
st.markdown("""
This application demonstrates the implementation of Linear Regression and Logistic Regression 
on a dataset. It evaluates model performance using Mean Squared Error (MSE) and R² score.
""")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv("mobile_usage_behavioral_analysis.csv")
    st.subheader("📂 Dataset Preview")
    st.dataframe(df)
    columns = df.columns.tolist()
    x_col = st.selectbox("Select Feature (X)", columns)
    y_col = st.selectbox("Select Target (Y)", columns)
    X = df[[x_col]]
    y = df[y_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    st.header("📈 Linear Regression")
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred = linear_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"**Mean Squared Error:** {mse}")
    st.write(f"**R² Score:** {r2}")
    st.line_chart(df)
    fig, ax = plt.subplots()
    ax.scatter(X_test, y_test, label="Actual")
    ax.plot(X_test, y_pred, color='red', label="Predicted")
    ax.legend()
    st.pyplot(fig)
    st.header("📊 Logistic Regression")
    if y.nunique() <= 10:
        log_model = LogisticRegression()
        log_model.fit(X_train, y_train)
        y_pred_log = log_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred_log)
        st.write(f"**Accuracy:** {acc}")
    else:
        st.warning("Target is not suitable for Logistic Regression (needs categorical data)")
