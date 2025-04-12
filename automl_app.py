# automl_app.py — Streamlit-based AutoML App

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

st.set_page_config(page_title="AutoML System", layout="wide")

st.title("🤖 AutoML: End-to-End ML Pipeline")

# File uploader
uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Raw Data Preview")
    st.dataframe(df.head())

    # Target selection
    target_col = st.selectbox("Select target column", df.columns)
    task_type = st.radio("Task Type", ["classification", "regression"])

    # Data Cleaning
    df.drop_duplicates(inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()

    # Feature separation
    categorical = df.select_dtypes(include='object').columns.tolist()
    numerical = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target_col in categorical: categorical.remove(target_col)
    if target_col in numerical: numerical.remove(target_col)

    # Preprocessor pipeline
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    preprocessor = ColumnTransformer([
        ("cat", cat_pipe, categorical),
        ("num", num_pipe, numerical)
    ])

    # Data splitting
    X = df.drop(columns=target_col)
    y = df[target_col]
    stratify = y if task_type == 'classification' and len(set(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=stratify)

    # Model selection
    MODELS = {
        'classification': [LogisticRegression(), RandomForestClassifier(), XGBClassifier(), LGBMClassifier()],
        'regression': [Ridge(), Lasso(), RandomForestRegressor(), XGBRegressor(), LGBMRegressor()]
    }

    st.subheader("⚙️ Model Performance")
    best_model, best_score = None, -np.inf if task_type == 'classification' else np.inf
    for model in MODELS[task_type]:
        pipe = Pipeline([
            ("pre", preprocessor),
            ("model", model)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        score = accuracy_score(y_test, y_pred) if task_type == 'classification' else mean_squared_error(y_test, y_pred, squared=False)
        st.write(f"{model.__class__.__name__} - Score: {score}")
        if (task_type == 'classification' and score > best_score) or \
           (task_type == 'regression' and score < best_score):
            best_model = pipe
            best_score = score

    # Save model
    with open("best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    st.success("✅ Best model saved as 'best_model.pkl'")

    # SHAP Explanation
    try:
        st.subheader("📈 Feature Importance (SHAP)")
        explainer = shap.Explainer(best_model.named_steps['model'], X_test)
        shap_values = explainer(X_test)
        fig = shap.plots.beeswarm(shap_values, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning("SHAP not supported for this model: " + str(e))

    # Prediction
    st.subheader("🔮 Make a Prediction")
    input_data = {col: st.text_input(f"{col}") for col in X.columns}
    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        input_df = input_df.apply(pd.to_numeric, errors='ignore')
        model = pickle.load(open("best_model.pkl", "rb"))
        prediction = model.predict(input_df)
        st.success(f"Prediction: {prediction[0]}")
