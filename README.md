## SmartPremium: Predicting Insurance Costs with Machine Learning

## Problem Statement

Insurance companies use factors such as age, income, health status, and claim history to estimate premiums for their customers.
The goal of this project is to develop a machine learning model that accurately predicts insurance premiums using customer and policy information, enabling data-driven pricing, risk assessment, and real-time quote estimation.

## Project Objectives

- Load and explore the insurance dataset.
- Perform data cleaning, normalization, and scaling.
- Develop regression models to predict insurance premiums.
- Build a complete preprocessing + modeling pipeline.
- Track experiments using MLflow (with DagsHub)
- Evaluate the final model on independent test data.
- Deploy the trained model as a Streamlit web application.

## Technologies Used

- Programming Language : Python
- Data Processing : Pandas, NumPy
- Feature Engineering : Box-Cox Transform, StandardScaler
- Machine Learning : Scikit-Learn, XGBoost
- Experiment Tracking : MLflow (DagsHub Tracking Server)
- Model Deployment : Streamlit Cloud
- Development Tools : VS Code, Jupyter Notebook

## Files & Notebooks

- 🔗 `model_building.ipynb`: for EDA, feature transformation, and model training
- 🔗 `mlpipeline_mlflow.ipynb`: for building pipelines, tracking with MLflow
- 🔗 `model_evaluate.ipynb`: for test-data evaluation
- 🔗 `preprocessing_pipeline.pkl`: serialized preprocessing pipeline
- 🔗 `smart_premium_model.pkl`: final production model

## Dataset Overview

- Format: CSV
- Records: 200,000
- Features: 20 customer, lifestyle, financial, and policy attributes
- Target Variable: Premium Amount
- [!Note]: This is a synthetic dataset created for educational and project use.

### Dataset characteristics:

- Missing values
- Incorrect data types
- Skewed numerical features
- Mixed data types: categorical, numerical, text, and date features

### 🔗 Dataset URL:

- [Google Drive Folder (Dataset)](https://drive.google.com/drive/folders/1GNSocgMntDHdTVmT2q0p1sE5iZss2h5_?usp=drive_link)

## Model Building Summary

In model_building.ipynb, the following steps were performed:

### ✔ Feature Selection & Cleaning

- Set the ID column as index.
- Dropped non-predictive features:
    - `"Customer Feedback"`
    - `"Policy Start Date"`

### ✔ Feature Transformation

- Applied Box-Cox transformation (λ = 0.5) to `"Annual Income"` to reduce skewness.

### ✔ Scaling

- Applied StandardScaler to all independent features to ensure uniform scaling.

### ✔ Model Training

- Trained multiple regression models: Linear Regression, Decision Tree Regressor, Random Forest Regressor, XGBoost Regressor.
- Models evaluated on test data using MAE, RMSE, and R² score.

## ML Pipeline & MLflow (DagsHub)

In `mlpipeline_mlflow.ipynb`:

### ✔ Preprocessing Pipeline

- Built a Scikit-Learn pipeline automating:
    - Column dropping
    - Box-Cox transformation
    - Standard scaling
- Saved as:
    - 📁 `preprocessing_pipeline.pkl`

### ✔ MLflow Tracking

- Connected MLflow to DagsHub tracking server.
- Logged all experiments: parameters, metrics, and artifacts.
- Best-performing model selected and promoted to Production.
- MLflow Tracking URL: [Click here](https://dagshub.com/nithis127/Smart_Premium.mlflow)

### ✔ Model Export

Final production model saved as:
- 📁 `smart_premium_model.pkl`

## Model Evaluation (Using Test Data)

- Applied preprocessing pipeline to the test set.
- Predictions generated using the production model.
- Metrics used for evaluation:
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - R² Score

## Streamlit Deployment

- Developed a Streamlit web app for real-time premium prediction.
- App loads both:
    - `preprocessing_pipeline.pkl`
    - `smart_premium_model.pkl`
- [Deployed on Streamlit Cloud](https://smartpremium-ibje7qsuzbkufzlkxfayub.streamlit.app/)

## Streamlit Application Screenshots

![image alt](https://github.com/nithis127/Smart_Premium/blob/e23c07d3f7bfc4575ad8befef604c78b63a24088/sp_streamlit_ss1.png)

![image alt](https://github.com/nithis127/Smart_Premium/blob/e23c07d3f7bfc4575ad8befef604c78b63a24088/sp_streamlit_ss2.png)

## Results

- Full ML pipeline created with preprocessing, training, and deployment.
- Experiments tracked using MLflow and DagsHub.
- Model evaluated on independent test data.
- Streamlit app deployed for public access.

## Project Evaluation Metrics

- Mean Absolute Error (MAE) – Measures average prediction error.
- Root Mean Squared Error (RMSE) – Measures overall prediction accuracy.
- R² Score – Measures variance explained by the model.

## Conclusion

The Smart_Premium project demonstrates a complete end-to-end ML workflow:
- Data preprocessing and transformation
- Feature scaling and Box-Cox normalization
- Model training and evaluation on test data
- MLflow experiment tracking (DagsHub)
- Production-ready model deployment with Streamlit Cloud
This project provides a scalable solution for real-time insurance premium prediction.
