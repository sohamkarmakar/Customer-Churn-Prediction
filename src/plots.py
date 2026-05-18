import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set standard styles and professional color palette
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Cohesive Palette: Deep Navy Blue for No Churn (0), Coral/Salmon for Churn (1)
CHURN_PALETTE = {0: '#1e3d59', 1: '#ff6e40'}
CHURN_LABELS = {0: 'Loyal Customer', 1: 'Churned'}

def set_style():
    """Set Seaborn aesthetic styles."""
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.1)

def plot_churn_distribution(df, output_dir):
    """
    1. Churn Distribution (bar chart showing imbalance)
    """
    set_style()
    plt.figure(figsize=(7, 6))
    
    churn_counts = df['Churn'].value_counts()
    churn_pct = df['Churn'].value_counts(normalize=True) * 100
    
    # Map binary target back to friendly names
    plot_df = df.copy()
    plot_df['Churn Status'] = plot_df['Churn'].map(CHURN_LABELS)
    
    ax = sns.countplot(
        x='Churn Status', 
        data=plot_df, 
        palette={'Loyal Customer': '#1e3d59', 'Churned': '#ff6e40'},
        hue='Churn Status',
        legend=False
    )
    
    # Annotate bars with counts and percentages
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        pct = churn_pct.iloc[i]
        ax.annotate(f'{int(height)}\n({pct:.1f}%)',
                    (p.get_x() + p.get_width() / 2., height / 2.0),
                    ha='center', va='center',
                    xytext=(0, 0), textcoords='offset points',
                    color='white', fontweight='bold', fontsize=12)
                    
    plt.title('Customer Churn Distribution (Class Imbalance)', pad=15, fontsize=14, fontweight='bold')
    plt.xlabel('Churn Status', labelpad=10)
    plt.ylabel('Number of Customers', labelpad=10)
    plt.tight_layout()
    
    path = os.path.join(output_dir, 'churn_distribution.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")

def plot_correlation_heatmap(df, output_dir):
    """
    2. Correlation Heatmap of numerical columns
    """
    set_style()
    plt.figure(figsize=(8, 7))
    
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Filter out customerID or other identifiers if they exist as numeric
    numeric_cols = [c for c in numeric_cols if c not in ['customerID', 'Has_Partner_and_Dependents']]
    
    corr = df[numeric_cols].corr()
    
    # Generate a mask for the upper triangle (cleaner display)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr, 
        mask=mask,
        annot=True, 
        cmap='coolwarm', 
        fmt=".2f", 
        linewidths=0.5,
        vmin=-1, vmax=1,
        cbar_kws={"shrink": .8}
    )
    
    plt.title('Correlation Matrix of Numerical Features', pad=15, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")

def plot_charges_vs_churn(df, output_dir):
    """
    3. Monthly Charges vs Churn (KDE Plot and Boxplot)
    """
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    plot_df = df.copy()
    plot_df['Churn Status'] = plot_df['Churn'].map(CHURN_LABELS)
    
    # KDE Plot
    sns.kdeplot(
        data=plot_df, 
        x='MonthlyCharges', 
        hue='Churn Status', 
        fill=True, 
        common_norm=False, 
        palette={'Loyal Customer': '#1e3d59', 'Churned': '#ff6e40'},
        alpha=0.6,
        linewidth=2,
        ax=axes[0]
    )
    axes[0].set_title('Monthly Charges Density Distribution by Churn', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Monthly Charges ($)')
    axes[0].set_ylabel('Density')
    
    # Boxplot
    sns.boxplot(
        x='Churn Status', 
        y='MonthlyCharges', 
        data=plot_df, 
        palette={'Loyal Customer': '#1e3d59', 'Churned': '#ff6e40'},
        hue='Churn Status',
        legend=False,
        ax=axes[1]
    )
    axes[1].set_title('Monthly Charges Range by Churn', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Churn Status')
    axes[1].set_ylabel('Monthly Charges ($)')
    
    plt.suptitle('Monthly Charges vs Customer Churn Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    path = os.path.join(output_dir, 'monthly_charges_vs_churn.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")

def plot_tenure_vs_churn(df, output_dir):
    """
    4. Tenure vs Churn (KDE Plot and Boxplot)
    """
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    plot_df = df.copy()
    plot_df['Churn Status'] = plot_df['Churn'].map(CHURN_LABELS)
    
    # KDE Plot
    sns.kdeplot(
        data=plot_df, 
        x='tenure', 
        hue='Churn Status', 
        fill=True, 
        common_norm=False, 
        palette={'Loyal Customer': '#1e3d59', 'Churned': '#ff6e40'},
        alpha=0.6,
        linewidth=2,
        ax=axes[0]
    )
    axes[0].set_title('Tenure Density Distribution by Churn', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Tenure (Months)')
    axes[0].set_ylabel('Density')
    
    # Boxplot
    sns.boxplot(
        x='Churn Status', 
        y='tenure', 
        data=plot_df, 
        palette={'Loyal Customer': '#1e3d59', 'Churned': '#ff6e40'},
        hue='Churn Status',
        legend=False,
        ax=axes[1]
    )
    axes[1].set_title('Tenure Range by Churn', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Churn Status')
    axes[1].set_ylabel('Tenure (Months)')
    
    plt.suptitle('Tenure vs Customer Churn Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    path = os.path.join(output_dir, 'tenure_vs_churn.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")

def plot_contract_analysis(df, output_dir):
    """
    5. Contract Type analysis vs Churn
    """
    set_style()
    plt.figure(figsize=(8, 6))
    
    # Compute proportions of churn per contract type
    contract_churn = df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack() * 100
    contract_churn.rename(columns=CHURN_LABELS, inplace=True)
    
    ax = contract_churn.plot(kind='bar', stacked=True, color=['#1e3d59', '#ff6e40'], edgecolor='black', figsize=(9, 6))
    
    # Annotate percentages in the stacked bars
    for container in ax.containers:
        labels = [f'{val:.1f}%' if val > 5 else '' for val in container.datavalues]
        ax.bar_label(container, labels=labels, label_type='center', color='white', fontweight='bold', fontsize=11)
        
    plt.title('Churn Rate Analysis by Contract Type', pad=15, fontsize=14, fontweight='bold')
    plt.xlabel('Contract Type', labelpad=10)
    plt.ylabel('Percentage (%)', labelpad=10)
    plt.xticks(rotation=0)
    plt.legend(title='Churn Status', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    path = os.path.join(output_dir, 'contract_type_vs_churn.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")

def plot_internet_service_analysis(df, output_dir):
    """
    6. Internet Service Type analysis vs Churn
    """
    set_style()
    plt.figure(figsize=(8, 6))
    
    internet_churn = df.groupby('InternetService')['Churn'].value_counts(normalize=True).unstack() * 100
    internet_churn.rename(columns=CHURN_LABELS, inplace=True)
    
    ax = internet_churn.plot(kind='bar', stacked=True, color=['#1e3d59', '#ff6e40'], edgecolor='black', figsize=(9, 6))
    
    for container in ax.containers:
        labels = [f'{val:.1f}%' if val > 5 else '' for val in container.datavalues]
        ax.bar_label(container, labels=labels, label_type='center', color='white', fontweight='bold', fontsize=11)
        
    plt.title('Churn Rate Analysis by Internet Service Type', pad=15, fontsize=14, fontweight='bold')
    plt.xlabel('Internet Service Type', labelpad=10)
    plt.ylabel('Percentage (%)', labelpad=10)
    plt.xticks(rotation=0)
    plt.legend(title='Churn Status', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    path = os.path.join(output_dir, 'internet_service_vs_churn.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")

def plot_demographics_analysis(df, output_dir):
    """
    7. Gender and Senior Citizen vs Churn analysis
    """
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    plot_df = df.copy()
    plot_df['Churn Status'] = plot_df['Churn'].map(CHURN_LABELS)
    
    # 7a. Gender vs Churn
    gender_churn = df.groupby('gender')['Churn'].value_counts(normalize=True).unstack() * 100
    gender_churn.rename(columns=CHURN_LABELS, inplace=True)
    gender_churn.plot(kind='bar', stacked=True, color=['#1e3d59', '#ff6e40'], edgecolor='black', ax=axes[0])
    axes[0].set_title('Churn Rate by Gender', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Gender')
    axes[0].set_ylabel('Percentage (%)')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    axes[0].get_legend().remove()
    
    # Annotate gender percentages
    for container in axes[0].containers:
        labels = [f'{val:.1f}%' if val > 5 else '' for val in container.datavalues]
        axes[0].bar_label(container, labels=labels, label_type='center', color='white', fontweight='bold', fontsize=11)
        
    # 7b. Senior Citizen vs Churn
    # SeniorCitizen is 0 or 1
    plot_df['Senior Citizen Status'] = plot_df['SeniorCitizen'].map({0: 'Non-Senior', 1: 'Senior Citizen'})
    senior_churn = plot_df.groupby('Senior Citizen Status')['Churn'].value_counts(normalize=True).unstack() * 100
    senior_churn.rename(columns=CHURN_LABELS, inplace=True)
    senior_churn.plot(kind='bar', stacked=True, color=['#1e3d59', '#ff6e40'], edgecolor='black', ax=axes[1])
    axes[1].set_title('Churn Rate by Senior Citizen Status', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Senior Citizen Status')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    
    # Annotate senior citizen percentages
    for container in axes[1].containers:
        labels = [f'{val:.1f}%' if val > 5 else '' for val in container.datavalues]
        axes[1].bar_label(container, labels=labels, label_type='center', color='white', fontweight='bold', fontsize=11)
        
    # Add unified legend
    axes[1].legend(title='Churn Status', loc='upper right')
    
    plt.suptitle('Demographics Churn Analysis: Gender vs Senior Citizen', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    path = os.path.join(output_dir, 'demographics_vs_churn.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")

def plot_feature_importance(importances, feature_names, output_path, top_n=15):
    """
    11a. Feature Importance visualization (horizontal bar plot)
    """
    set_style()
    plt.figure(figsize=(10, 7))
    
    # Combine feature names and importances
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(top_n)
    
    sns.barplot(
        x='Importance',
        y='Feature',
        data=feat_imp,
        hue='Feature',
        palette='viridis',
        legend=False
    )
    
    plt.title(f'Top {top_n} Features by Importance (Best Model)', pad=15, fontsize=14, fontweight='bold')
    plt.xlabel('Relative Importance Score', labelpad=10)
    plt.ylabel('Feature Name', labelpad=10)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved feature importance plot to {output_path}")

def plot_model_comparison(comparison_df, output_path):
    """
    11b. Model Performance Comparison Graph (bar chart comparing key metrics)
    """
    set_style()
    plt.figure(figsize=(10, 6))
    
    # Melt comparison_df for plotting with Seaborn
    # Columns of comparison_df are: Model, Accuracy, Precision, Recall, F1-Score, ROC-AUC
    melted_df = pd.melt(
        comparison_df, 
        id_vars=['Model'], 
        value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        var_name='Metric', 
        value_name='Value'
    )
    
    ax = sns.barplot(
        x='Model',
        y='Value',
        hue='Metric',
        data=melted_df,
        palette='Set2'
    )
    
    # Add values on top of bars
    for container in ax.containers:
        # Draw labels only for large bars
        labels = [f'{val:.2f}' for val in container.datavalues]
        ax.bar_label(container, labels=labels, label_type='edge', fontsize=8, padding=3)
        
    plt.title('Machine Learning Models Performance Comparison', pad=15, fontsize=14, fontweight='bold')
    plt.xlabel('Machine Learning Model', labelpad=10)
    plt.ylabel('Metric Score (0.0 - 1.0)', labelpad=10)
    plt.ylim(0.0, 1.15)  # Make room for text labels
    plt.legend(title='Evaluation Metric', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved model comparison plot to {output_path}")

def generate_all_eda_plots(df, output_dir):
    """
    Generate and save all 7 core EDA visualizations inside the output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating EDA plots in directory: {output_dir}")
    
    # Call each EDA plotting function
    plot_churn_distribution(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_charges_vs_churn(df, output_dir)
    plot_tenure_vs_churn(df, output_dir)
    plot_contract_analysis(df, output_dir)
    plot_internet_service_analysis(df, output_dir)
    plot_demographics_analysis(df, output_dir)
    
    print("All EDA visualizations successfully generated and saved!")
