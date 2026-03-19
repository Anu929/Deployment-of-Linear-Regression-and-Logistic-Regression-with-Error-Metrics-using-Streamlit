import os
os.system('pip install scikit-learn')
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

st.title("📊 Mobile Usage Behavioral Analysis")
st.write("Linear Regression & Logistic Regression with Error Metrics")

# Upload dataset
file = st.file_uploader("Upload CSV file", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # Select features and target
    columns = df.columns.tolist()

    feature_cols = st.multiselect("Select Feature Columns", columns)
    target_col = st.selectbox("Select Target Column", columns)

    if feature_cols and target_col:

        X = df[feature_cols]
        y = df[target_col]

        # Convert categorical to numeric
        X = pd.get_dummies(X)
        if y.dtype == 'object':
            y = pd.factorize(y)[0]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # ------------------ LINEAR REGRESSION ------------------
        st.subheader("📈 Linear Regression")

        lr = LinearRegression()
        lr.fit(X_train, y_train)

        y_pred_lr = lr.predict(X_test)

        mse_lr = mean_squared_error(y_test, y_pred_lr)
        rmse_lr = np.sqrt(mse_lr)

        st.write("Mean Squared Error (MSE):", mse_lr)
        st.write("Root Mean Squared Error (RMSE):", rmse_lr)

        # ------------------ LOGISTIC REGRESSION ------------------
        st.subheader("📉 Logistic Regression")

        # Convert target to binary if needed
        y_binary = (y > y.mean()).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.2, random_state=42
        )

        logr = LogisticRegression(max_iter=1000)
        logr.fit(X_train, y_train)

        y_pred_log = logr.predict(X_test)

        mse_log = mean_squared_error(y_test, y_pred_log)
        rmse_log = np.sqrt(mse_log)
        acc = accuracy_score(y_test, y_pred_log)

        st.write("Accuracy:", acc)
        st.write("Mean Squared Error (MSE):", mse_log)
        st.write("Root Mean Squared Error (RMSE):", rmse_log)
