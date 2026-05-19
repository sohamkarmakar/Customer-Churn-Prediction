import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_raw_data(data_path):
    """
    Load raw CSV data from path.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Raw data file not found at {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset successfully loaded from {data_path}. Shape: {df.shape}")
    return df

def clean_data(df):
    """
    Perform core data cleaning operations:
    1. Strip white spaces from all string columns
    2. Convert TotalCharges to numeric, handling empty spaces as NaN and imputing
    3. Drop duplicates
    4. Convert Target 'Churn' to binary integer (Yes: 1, No: 0)
    """
    # Create a copy to prevent SettingWithCopyWarning
    cleaned_df = df.copy()
    
    # 1. Strip whitespaces
    for col in cleaned_df.select_dtypes(include='object').columns:
        cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
        
    # 2. TotalCharges has blank spaces (" ") for new customers (tenure = 0).
    # Since tenure is 0, they haven't paid anything yet, so we replace with NaN and fill with 0.0.
    # TotalCharges is originally read as object/string due to these spaces.
    cleaned_df['TotalCharges'] = cleaned_df['TotalCharges'].replace('', np.nan)
    cleaned_df['TotalCharges'] = pd.to_numeric(cleaned_df['TotalCharges'], errors='coerce')
    
    # Fill missing TotalCharges with 0.0 (logically, tenure=0 means 0 total charges)
    cleaned_df['TotalCharges'] = cleaned_df['TotalCharges'].fillna(0.0)
    
    # 3. Duplicate removal
    duplicates_count = cleaned_df.duplicated().sum()
    if duplicates_count > 0:
        print(f"Found {duplicates_count} duplicate rows. Removing them...")
        cleaned_df.drop_duplicates(inplace=True)
        
    # Drop customerID for modeling, but keep it if we need to track (handled at training time)
    
    # 4. Target encoding
    if 'Churn' in cleaned_df.columns:
        cleaned_df['Churn'] = cleaned_df['Churn'].map({'Yes': 1, 'No': 0})
        
    return cleaned_df

def engineer_features(df):
    """
    Engineer domain-specific features helpful for customer churn prediction:
    1. tenure_group: Bin the tenure into standard contract cycles
    2. Number_of_Services: Total online services a customer uses
    3. Has_Partner_and_Dependents: Interaction feature for family status
    4. Monthly_to_Total_Ratio: Relationship between monthly and total charges
    """
    fe_df = df.copy()
    
    # 1. Binning Tenure
    # Binning numerical variables can capture non-linear relationships.
    def get_tenure_group(months):
        if months <= 12:
            return '0-1 Year'
        elif months <= 24:
            return '1-2 Years'
        elif months <= 48:
            return '2-4 Years'
        elif months <= 60:
            return '4-5 Years'
        else:
            return 'Over 5 Years'
            
    fe_df['tenure_group'] = fe_df['tenure'].apply(get_tenure_group)
    
    # 2. Total services used
    # List of online services offered by Telco
    services = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    # Count how many of these services are 'Yes'
    service_count = np.zeros(len(fe_df))
    for service in services:
        if service in fe_df.columns:
            service_count += (fe_df[service] == 'Yes').astype(int)
            
    fe_df['Number_of_Services'] = service_count
    
    # 3. Interaction Feature: Partner and Dependents (family strength)
    if 'Partner' in fe_df.columns and 'Dependents' in fe_df.columns:
        fe_df['Has_Partner_and_Dependents'] = ((fe_df['Partner'] == 'Yes') & (fe_df['Dependents'] == 'Yes')).astype(int)
    else:
        fe_df['Has_Partner_and_Dependents'] = 0
        
    # 4. Monthly charge ratio compared to total charge
    # Adding a small epsilon 1e-5 to avoid division by zero
    fe_df['Monthly_to_Total_Ratio'] = fe_df['MonthlyCharges'] / (fe_df['TotalCharges'] + 1e-5)
    
    return fe_df

def get_preprocessing_pipeline(numerical_cols, categorical_cols):
    """
    Construct the Scikit-learn preprocessing ColumnTransformer pipeline.
    Numerical: Impute with median, scale with StandardScaler
    Categorical: Impute with mode, encode with OneHotEncoder (dropping first category to avoid dummy variable trap)
    """
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ], remainder='drop')
    
    return preprocessor

def preprocess_full_dataset(raw_data_path, processed_dir_path):
    """
    End-to-end preprocessing orchestrator. Loads raw data, cleans it, engineers features,
    splits into train/test, and saves the cleaned datasets.
    """
    os.makedirs(processed_dir_path, exist_ok=True)
    
    # Load and clean
    df = load_raw_data(raw_data_path)
    cleaned_df = clean_data(df)
    
    # Feature engineering
    engineered_df = engineer_features(cleaned_df)
    
    # Separate features and target
    if 'Churn' in engineered_df.columns:
        X = engineered_df.drop(columns=['Churn', 'customerID'], errors='ignore')
        y = engineered_df['Churn']
    else:
        X = engineered_df.drop(columns=['customerID'], errors='ignore')
        y = None
        
    # Train-test split (stratified on y to preserve target balance)
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Save processed splits
        train_df = X_train.copy()
        train_df['Churn'] = y_train
        test_df = X_test.copy()
        test_df['Churn'] = y_test
        
        train_df.to_csv(os.path.join(processed_dir_path, 'churn_train.csv'), index=False)
        test_df.to_csv(os.path.join(processed_dir_path, 'churn_test.csv'), index=False)
        
        print(f"Train split saved to {os.path.join(processed_dir_path, 'churn_train.csv')} with shape: {train_df.shape}")
        print(f"Test split saved to {os.path.join(processed_dir_path, 'churn_test.csv')} with shape: {test_df.shape}")
        
        return X_train, X_test, y_train, y_test
    else:
        # Saving full processed data (without target)
        full_processed_path = os.path.join(processed_dir_path, 'churn_processed.csv')
        engineered_df.to_csv(full_processed_path, index=False)
        print(f"Processed dataset saved to {full_processed_path}")
        return engineered_df
