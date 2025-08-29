import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    r2_score, mean_absolute_error, mean_squared_error
)


# App title
st.set_page_config(page_title="CSV ML Trainer", layout="centered")
st.title("📊 ML Model ")

#Upload CSV File
uploaded_file = st.file_uploader("📁 Upload your CSV file (only .csv allowed)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📝 Dataset Preview")
    st.dataframe(df.head())

    #  Button to check and show null values
    if st.button("🔍 Check for Null Values"):
        nulls = df.isnull().sum()
        if nulls.sum() == 0:
            st.success("✅ No Null Values Found in the Dataset.")
        else:
            st.warning("⚠️ Null Values Detected!")
            st.dataframe(nulls[nulls > 0].to_frame(name='Null Count'))
    
    # for fill null values 
    st.subheader("🛠️ Handling Missing Values with SimpleImputer")
    imputer = SimpleImputer(strategy='most_frequent')  # or use 'mean' for numeric data
    df[df.columns] = imputer.fit_transform(df)
    st.success("✅ Null values filled using SimpleImputer (strategy='most_frequent').")
    st.dataframe(df.head())

    # Drop rows with missing values for simplicity
    df = df.dropna()
    st.info(f"🧹 Rows with null values dropped. Remaining rows: {len(df)}")
    
    

    # Convert Categorical to Numerical
    st.subheader("🔄 Categorical to Numerical Conversion")
    label_encoders = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    st.success("✅ Converted all categorical columns to numerical.")
    st.dataframe(df.head())

    # Target Column Selection
    target_column = st.selectbox("🎯 Select the Target Column", df.columns)

    if target_column:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.subheader("📊 Train-Test Split Summary")
        st.write(f"Training Set: {X_train.shape}")
        st.write(f"Testing Set: {X_test.shape}")

        # Model Selection
        models = {
            "Linear Regression": LinearRegression(),
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "DecisionTreeRegressor": DecisionTreeRegressor(),
            "Random Forest": RandomForestClassifier(),
            "RandomForestRegressor": RandomForestRegressor(),
            "Support Vector Machine": SVC(),
            "Naive Bayes": GaussianNB()
        }

        selected_model = st.selectbox("🤖 Select a Machine Learning Model", list(models.keys()))

        if selected_model:
            model = models[selected_model]
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            st.subheader("📈 Available Evaluation Metrics")

            # Determine if it's classification or regression
            is_classification = "Classification" in selected_model

            pem_options = []

            if is_classification:
                acc = accuracy_score(y_test, predictions)
                prec = precision_score(y_test, predictions, average="macro", zero_division=0)
                rec = recall_score(y_test, predictions, average="macro", zero_division=0)
                f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
                conf_matrix = confusion_matrix(y_test, predictions)
                class_report = classification_report(y_test, predictions, output_dict=True)

                pem_options = {
                    "Accuracy": acc,
                    "Precision (Macro Avg)": prec,
                    "Recall (Macro Avg)": rec,
                    "F1-Score (Macro Avg)": f1,
                    "Confusion Matrix": conf_matrix,
                    "Classification Report": class_report
                }

            else:
                r2 = r2_score(y_test, predictions)
                mae = mean_absolute_error(y_test, predictions)
                mse = mean_squared_error(y_test, predictions)
                rmse = np.sqrt(mse)

                pem_options = {
                    "R² Score": r2,
                    "MAE (Mean Absolute Error)": mae,
                    "MSE (Mean Squared Error)": mse,
                    "RMSE (Root Mean Squared Error)": rmse
                }

            selected_pem = st.selectbox("📊 Select a Performance Evaluation Metric", list(pem_options.keys()))

            #  Show selected PEM
            st.subheader("🧾 PEM Output")

            if selected_pem:
                pem_value = pem_options[selected_pem]

                if isinstance(pem_value, (float, int)):
                    st.success(f"📌 {selected_pem}: **{pem_value:.4f}**")
                elif isinstance(pem_value, np.ndarray):
                    st.write(f"📌 {selected_pem}:")
                    st.dataframe(pem_value)
                elif isinstance(pem_value, dict):
                    st.write(f"📌 {selected_pem}:")
                    st.json(pem_value)
                else:
                    st.write(pem_value)