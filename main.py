import os
import argparse
import pandas as pd
from src.preprocess import preprocess_full_dataset, clean_data, engineer_features, load_raw_data
from src.plots import generate_all_eda_plots
from src.train import train_and_evaluate_all

# Define default paths relative to workspace root
RAW_DATA_PATH = os.path.join('data', 'raw', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
PROCESSED_DIR = os.path.join('data', 'processed')
MODELS_DIR = os.path.join('models')
OUTPUTS_DIR = os.path.join('outputs')

def run_eda_pipeline():
    """
    Load data and generate all EDA plots under outputs/plots/
    """
    print("\n==============================================")
    print("STEP 1: Starting Exploratory Data Analysis (EDA)")
    print("==============================================")
    
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw data file not found at {RAW_DATA_PATH}. Please ensure it is placed there.")
        
    df = load_raw_data(RAW_DATA_PATH)
    
    # Run a light clean (convert TotalCharges, map Churn target) to make EDA accurate
    cleaned_df = clean_data(df)
    
    # Plotting directory
    plots_dir = os.path.join(OUTPUTS_DIR, 'plots')
    generate_all_eda_plots(cleaned_df, plots_dir)
    print(f"EDA Generation complete! All 7 plots saved in: {plots_dir}")

def run_training_pipeline():
    """
    Preprocess data, split it, train models, tune best model, and save outputs.
    """
    print("\n==============================================")
    print("STEP 2: Data Preprocessing & Model Training")
    print("==============================================")
    
    # 1. End-to-end preprocessing
    print("Starting full data cleaning and feature engineering split...")
    X_train, X_test, y_train, y_test = preprocess_full_dataset(RAW_DATA_PATH, PROCESSED_DIR)
    
    # 2. Train, evaluate, tune, and save best model
    print("Starting model training, cross-validation, and tuning...")
    model_bundle = train_and_evaluate_all(PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR)
    
    print("\n==============================================")
    print("Model Training & Evaluation Complete!")
    print(f"Serialized Best Model: {model_bundle['model_name']}")
    print(f"F1-Score: {model_bundle['metrics']['F1-Score']:.4f}")
    print(f"ROC-AUC: {model_bundle['metrics']['ROC-AUC']:.4f}")
    print("==============================================")

def main():
    parser = argparse.ArgumentParser(
        description="End-to-End Telco Customer Churn Prediction ML Pipeline",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--eda', '-e',
        action='store_true',
        help='Run only the Exploratory Data Analysis (EDA) and generate visual plots'
    )
    group.add_argument(
        '--train', '-t',
        action='store_true',
        help='Run data preprocessing, feature engineering, and model training/tuning'
    )
    group.add_argument(
        '--all', '-a',
        action='store_true',
        help='Run entire pipeline end-to-end (EDA + Preprocessing + Training + Tuning)'
    )
    
    args = parser.parse_args()
    
    # Set standard directory structure if not exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUTS_DIR, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUTS_DIR, 'confusion_matrices'), exist_ok=True)
    
    # If no arguments provided, run both EDA and Training by default
    if not (args.eda or args.train or args.all):
        print("No execution mode specified. Running full end-to-end pipeline by default.")
        run_eda_pipeline()
        run_training_pipeline()
    else:
        if args.eda:
            run_eda_pipeline()
        elif args.train:
            run_training_pipeline()
        elif args.all:
            run_eda_pipeline()
            run_training_pipeline()

if __name__ == '__main__':
    main()
