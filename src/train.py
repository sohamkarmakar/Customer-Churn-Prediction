import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

# Import XGBoost if available, fallback gracefully
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: xgboost is not installed. XGBoost model will be skipped or simulated.")

from src.preprocess import get_preprocessing_pipeline
from src.plots import plot_feature_importance, plot_model_comparison

def save_confusion_matrix(y_true, y_pred, model_name, output_dir):
    """
    Generate and save a clean, professional confusion matrix heatmap.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        cbar=False,
        xticklabels=['Loyal', 'Churned'],
        yticklabels=['Loyal', 'Churned']
    )
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=12, fontweight='bold', pad=10)
    plt.ylabel('Actual Churn Status', fontsize=10)
    plt.xlabel('Predicted Churn Status', fontsize=10)
    plt.tight_layout()
    
    path = os.path.join(output_dir, f'cm_{model_name.lower().replace(" ", "_")}.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix: {path}")

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a model on standard classification metrics and return a dictionary.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.5
    
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    }
    
    return metrics, y_pred

def train_and_evaluate_all(processed_dir_path, models_dir_path, outputs_dir_path):
    """
    Main training and tuning pipeline. Fits preprocessor, trains 4 models,
    tunes the best one using GridSearchCV, generates final comparison graphs/tables,
    and serializes the best model bundle.
    """
    os.makedirs(models_dir_path, exist_ok=True)
    os.makedirs(outputs_dir_path, exist_ok=True)
    
    # 1. Load Processed Datasets
    train_path = os.path.join(processed_dir_path, 'churn_train.csv')
    test_path = os.path.join(processed_dir_path, 'churn_test.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Processed splits not found. Please run preprocessing first.")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=['Churn'])
    y_train = train_df['Churn']
    X_test = test_df.drop(columns=['Churn'])
    y_test = test_df['Churn']
    
    # 2. Identify Column Types for Preprocessor
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Number_of_Services', 'Monthly_to_Total_Ratio']
    
    # Categorical columns are all columns excluding numerical ones
    categorical_cols = [col for col in X_train.columns if col not in numerical_cols]
    
    print(f"Numerical Columns ({len(numerical_cols)}): {numerical_cols}")
    print(f"Categorical Columns ({len(categorical_cols)}): {categorical_cols}")
    
    # 3. Fit Preprocessing Pipeline
    preprocessor = get_preprocessing_pipeline(numerical_cols, categorical_cols)
    
    print("Fitting preprocessing pipeline on training data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get column names after transformation for Feature Importance analysis
    # Use fallback if get_feature_names_out is not supported (old sklearn versions)
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        # Construct feature names manually in worst case
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features = cat_encoder.get_feature_names_out(categorical_cols)
        feature_names = np.concatenate([numerical_cols, cat_features])
        
    print(f"Total features after One-Hot Encoding: {len(feature_names)}")
    
    # 4. Initialize Models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, class_weight='balanced')
    }
    
    if XGB_AVAILABLE:
        # Scale pos weight handles class imbalance in XGBoost
        scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
        models['XGBoost'] = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )
    else:
        print("XGBoost is not installed. Skipping XGBoost model.")
        
    # 5. Train and Evaluate Baseline Models
    results_list = []
    trained_models = {}
    
    cm_dir = os.path.join(outputs_dir_path, 'confusion_matrices')
    
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train_processed, y_train)
        trained_models[model_name] = model
        
        # Evaluate
        metrics, y_pred = evaluate_model(model, X_test_processed, y_test, model_name)
        results_list.append(metrics)
        
        # Save confusion matrix
        save_confusion_matrix(y_test, y_pred, model_name, cm_dir)
        
    # 6. Hyperparameter Tuning on Best Model
    # We will tune XGBoost if available, otherwise Random Forest
    tuned_model_name = "XGBoost (Tuned)" if XGB_AVAILABLE else "Random Forest (Tuned)"
    print(f"--- Running Hyperparameter Tuning for: {tuned_model_name} ---")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    if XGB_AVAILABLE:
        scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
        param_grid = {
            'max_depth': [3, 5, 7],
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0]
        }
        grid_search = GridSearchCV(
            estimator=XGBClassifier(
                use_label_encoder=False, 
                eval_metric='logloss', 
                scale_pos_weight=scale_pos_weight, 
                random_state=42
            ),
            param_grid=param_grid,
            scoring='f1', # Optimize for F1 score due to class imbalance
            cv=cv,
            n_jobs=-1,
            verbose=1
        )
    else:
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [6, 8, 10],
            'min_samples_split': [2, 5, 10],
            'class_weight': ['balanced', None]
        }
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            scoring='f1',
            cv=cv,
            n_jobs=-1,
            verbose=1
        )
        
    grid_search.fit(X_train_processed, y_train)
    best_estimator = grid_search.best_estimator_
    
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    
    # Evaluate Tuned Model
    metrics_tuned, y_pred_tuned = evaluate_model(best_estimator, X_test_processed, y_test, tuned_model_name)
    results_list.append(metrics_tuned)
    save_confusion_matrix(y_test, y_pred_tuned, tuned_model_name, cm_dir)
    trained_models[tuned_model_name] = best_estimator
    
    # 7. Create Performance Comparison Table
    comparison_df = pd.DataFrame(results_list)
    comparison_path = os.path.join(outputs_dir_path, 'model_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Saved model comparison table to {comparison_path}")
    print("\n--- Final Model Performance Comparison Table ---")
    print(comparison_df.to_string(index=False))
    
    # 8. Save Comparison Graph
    plot_model_comparison(comparison_df, os.path.join(outputs_dir_path, 'plots', 'model_comparison_graph.png'))
    
    # 9. Best Model Selection & Feature Importance
    # We select based on F1-Score because it balances Precision and Recall for class imbalance
    best_idx = comparison_df['F1-Score'].idxmax()
    best_row = comparison_df.iloc[best_idx]
    best_model_key = best_row['Model']
    final_best_model = trained_models[best_model_key]
    
    print(f"\n>>>> Best Model Selected based on F1-Score: {best_model_key} (F1: {best_row['F1-Score']:.4f}, ROC-AUC: {best_row['ROC-AUC']:.4f}) <<<<")
    
    # Feature Importance Plot
    importance_path = os.path.join(outputs_dir_path, 'plots', 'feature_importance.png')
    
    if hasattr(final_best_model, 'feature_importances_'):
        importances = final_best_model.feature_importances_
        plot_feature_importance(importances, feature_names, importance_path)
    elif hasattr(final_best_model, 'coef_'):
        # For Logistic Regression, coefficients act as feature importance
        importances = np.abs(final_best_model.coef_[0])
        plot_feature_importance(importances, feature_names, importance_path)
    else:
        print("Feature importances not directly extractable from the selected best model.")
        
    # 10. Model Bundle Serialization
    # Serialize model, preprocessor, and feature details together
    model_bundle = {
        'model': final_best_model,
        'model_name': best_model_key,
        'preprocessor': preprocessor,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'feature_names': list(feature_names),
        'metrics': best_row.to_dict()
    }
    
    bundle_path = os.path.join(models_dir_path, 'best_model.pkl')
    joblib.dump(model_bundle, bundle_path)
    print(f"Successfully saved best model bundle to: {bundle_path}")
    
    return model_bundle
