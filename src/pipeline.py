import os
import joblib
import pandas as pd
import numpy as np
from src.preprocess import clean_data, engineer_features

class ChurnPredictionPipeline:
    """
    Unified end-to-end inference pipeline.
    Loads the serialized model bundle and handles preprocessing, feature engineering,
    and model scoring for single customer dictionaries or bulk customer DataFrames.
    """
    def __init__(self, model_bundle_path='models/best_model.pkl'):
        if not os.path.exists(model_bundle_path):
            raise FileNotFoundError(f"Model bundle not found at {model_bundle_path}. Please train a model first.")
            
        print(f"Loading best model bundle from {model_bundle_path}...")
        self.bundle = joblib.load(model_bundle_path)
        self.model = self.bundle['model']
        self.preprocessor = self.bundle['preprocessor']
        self.numerical_cols = self.bundle['numerical_cols']
        self.categorical_cols = self.bundle['categorical_cols']
        self.feature_names = self.bundle['feature_names']
        self.model_name = self.bundle['model_name']
        print(f"Model successfully loaded. Best Model type: {self.model_name}")

    def _prepare_data(self, raw_df):
        """
        Apply identical data cleaning and feature engineering to the raw input.
        """
        # Ensure all columns required for cleaning and engineering are present
        # In single prediction, we fill missing columns with default/logical values
        prepared_df = raw_df.copy()
        
        # Apply standard cleaning steps (safely)
        if 'TotalCharges' in prepared_df.columns:
            prepared_df['TotalCharges'] = prepared_df['TotalCharges'].replace('', np.nan)
            prepared_df['TotalCharges'] = pd.to_numeric(prepared_df['TotalCharges'], errors='coerce')
            prepared_df['TotalCharges'] = prepared_df['TotalCharges'].fillna(0.0)
            
        # Strip whitespaces from string columns
        for col in prepared_df.select_dtypes(include='object').columns:
            prepared_df[col] = prepared_df[col].astype(str).str.strip()
            
        # Apply feature engineering
        engineered_df = engineer_features(prepared_df)
        
        # Ensure correct column ordering matching ColumnTransformer expected columns
        # (ColumnTransformer expects columns in specific orders or matches by name if DF)
        return engineered_df

    def predict_single(self, customer_dict):
        """
        Predict churn risk for a single customer (dictionary of raw feature values).
        Returns a rich summary with churn probability, confidence, and business recommendations.
        """
        # Convert dictionary to 1-row DataFrame
        raw_df = pd.DataFrame([customer_dict])
        
        # Clean and engineer features
        processed_df = self._prepare_data(raw_df)
        
        # Drop columns not used by preprocessor (like Churn, customerID)
        X = processed_df.drop(columns=['Churn', 'customerID'], errors='ignore')
        
        # Transform features using fitted preprocessor
        X_processed = self.preprocessor.transform(X)
        
        # Run prediction
        prob = self.model.predict_proba(X_processed)[0, 1]
        pred = self.model.predict(X_processed)[0]
        
        # Determine confidence level
        # TCS Interview Concept: Standard threshold is 0.5. We categorize risk bands
        if prob < 0.3:
            risk_level = "Low Risk"
            confidence = (1 - prob) * 100
        elif prob < 0.6:
            risk_level = "Medium Risk"
            confidence = max(prob, 1 - prob) * 100
        else:
            risk_level = "High Risk"
            confidence = prob * 100
            
        # Generate actionable business recommendations (fresher interview highlight)
        recommendations = []
        if risk_level in ["Medium Risk", "High Risk"]:
            if customer_dict.get('Contract') == 'Month-to-month':
                recommendations.append("Offer a discounted 1-Year or 2-Year contract upgrade to secure commitment.")
            if customer_dict.get('InternetService') == 'Fiber optic' and customer_dict.get('TechSupport') == 'No':
                recommendations.append("Provide a free month of Premium Tech Support to assist with fiber speed issues.")
            if customer_dict.get('OnlineSecurity') == 'No':
                recommendations.append("Recommend adding Online Security to protect their account and build loyalty.")
            if float(customer_dict.get('MonthlyCharges', 0)) > 80:
                recommendations.append("Propose a tailored value plan or bundle to reduce their monthly financial burden.")
        else:
            recommendations.append("Customer is loyal. Enroll in proactive loyalty reward programs to encourage advocacy.")
            
        if len(recommendations) == 0:
            recommendations.append("Continue standard quality support and monitor billing cycles.")
            
        return {
            'churn_prediction': int(pred),
            'churn_label': 'Churn Risk' if pred == 1 else 'Loyal Customer',
            'churn_probability': float(prob),
            'risk_level': risk_level,
            'confidence_score': float(confidence),
            'business_recommendations': recommendations
        }

    def predict_bulk(self, raw_df):
        """
        Predict churn risk for a batch of customers (Pandas DataFrame).
        Returns the original DataFrame with churn predictions and probabilities appended.
        """
        df_copy = raw_df.copy()
        
        # Preprocess bulk data
        processed_df = self._prepare_data(df_copy)
        
        # Drop columns not used by model
        X = processed_df.drop(columns=['Churn', 'customerID'], errors='ignore')
        
        # Transform features
        X_processed = self.preprocessor.transform(X)
        
        # Predict
        probs = self.model.predict_proba(X_processed)[:, 1]
        preds = self.model.predict(X_processed)
        
        # Append to copy
        df_copy['Churn_Probability'] = probs
        df_copy['Churn_Prediction'] = preds
        df_copy['Risk_Segment'] = pd.cut(
            probs, 
            bins=[0.0, 0.3, 0.6, 1.0], 
            labels=['Low Risk', 'Medium Risk', 'High Risk'],
            include_lowest=True
        )
        
        return df_copy
