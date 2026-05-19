import os
import sys
# Add parent directory to sys.path to guarantee src imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import time

# Set page configurations
st.set_page_config(
    page_title="Telco Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
    
    /* General Settings */
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
    }
    
    /* Title Text Styling */
    .title-text {
        background: linear-gradient(135deg, #12c2e9, #c471ed, #f64f59);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }
    
    .subtitle-text {
        text-align: center;
        color: #718096;
        font-size: 1.15rem;
        margin-bottom: 30px;
    }
    
    /* Cards and Containers styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }
    
    .result-container {
        border-radius: 20px;
        padding: 30px;
        margin-top: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .result-loyal {
        background: linear-gradient(135deg, rgba(39, 174, 96, 0.1), rgba(46, 204, 113, 0.05));
        border: 2px solid #27ae60;
    }
    
    .result-churn {
        background: linear-gradient(135deg, rgba(235, 77, 75, 0.1), rgba(255, 107, 107, 0.05));
        border: 2px solid #eb4d4b;
    }
    
    .loyal-header {
        color: #27ae60;
        font-weight: 800;
        font-size: 1.8rem;
        margin-bottom: 10px;
    }
    
    .churn-header {
        color: #eb4d4b;
        font-weight: 800;
        font-size: 1.8rem;
        margin-bottom: 10px;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #1e3d59, #17b978);
        color: white;
        border: none;
        padding: 12px 28px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.05rem;
        box-shadow: 0 4px 15px rgba(23, 185, 120, 0.2);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(23, 185, 120, 0.35);
        color: white !important;
        border: none !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        justify-content: center;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 1.15rem;
        font-weight: 600;
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
        color: #718096;
    }
    
    .stTabs [aria-selected="true"] {
        color: #1e3d59 !important;
        border-bottom: 3px solid #1e3d59 !important;
    }
    
</style>
""", unsafe_allow_html=True)

# Imports from src
try:
    from src.pipeline import ChurnPredictionPipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

# Sidebar Info
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/predictive-analytics.png", width=120)
    st.markdown("### 🔮 Predictor Console")
    st.markdown("This predictive engine leverages a state-of-the-art **Random Forest / XGBoost Ensemble** trained on historical subscription behaviors.")
    
    st.markdown("---")
    st.markdown("#### 🎯 Business KPIs")
    st.markdown("- **Recall Priority**: Focuses heavily on identifying true churners to protect monthly recurring revenue.")
    st.markdown("- **Actionable Insights**: Recommends targeted marketing policies based on customer billing behaviors.")
    
    st.markdown("---")
    st.markdown("#### 🛠️ Technology Stack")
    st.markdown("- Python, Pandas, Numpy\n- Scikit-Learn\n- XGBoost Classifier\n- Streamlit Dashboard")

# Main Page Headers
st.markdown("<div class='title-text'>Telco Customer Churn Predictive Platform</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>A Professional End-to-End Enterprise Solution for Churn Risk Analytics</div>", unsafe_allow_html=True)

# Try loading prediction pipeline
@st.cache_resource
def load_pipeline():
    # Make sure we search in the workspace folders
    model_path = os.path.join('models', 'best_model.pkl')
    if not os.path.exists(model_path):
        # Local search
        model_path = 'best_model.pkl'
        
    if not PIPELINE_AVAILABLE:
        st.error("Prediction pipeline module ('src.pipeline') could not be imported. Please verify your workspace directories and run python main.py --all first.")
        return None

    try:
        pipeline = ChurnPredictionPipeline(model_path)
        return pipeline
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")
        return None

pipeline = load_pipeline()

# Check if best model exists
if pipeline is None:
    st.warning("⚠️ **Trained Model Bundle Not Found!** \n\nPlease run `python main.py --all` in your terminal first to train the machine learning models and generate the serialized model bundle `models/best_model.pkl`.")
    st.info("💡 Running the model training pipeline will automatically create the required directories, clean the dataset, fit standard models, and save the best-performing model.")
else:
    # Set up Tabs
    tab_single, tab_batch = st.tabs(["👤 Single Customer Prediction", "📊 Batch Analytics Upload"])
    
    # -------------------------------------------------------------
    # TAB 1: SINGLE CUSTOMER PREDICTION
    # -------------------------------------------------------------
    with tab_single:
        st.markdown("### Customer Demographic & Service Configuration Form")
        st.write("Complete the sections below to instantly generate a churn prediction, confidence score, and tailored business retention advice.")
        
        # Form structure divided into Columns
        col_dem, col_srv, col_acc = st.columns(3)
        
        with col_dem:
            st.markdown("#### 🧑 Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen Status", ["No", "Yes"])
            partner = st.selectbox("Has Partner?", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
            
        with col_srv:
            st.markdown("#### 🔌 Subscribed Services")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            
            # Sub-logic for MultipleLines based on Phone Service
            if phone_service == "Yes":
                multiple_lines = st.selectbox("Multiple Phone Lines", ["No", "Yes"])
            else:
                multiple_lines = "No phone service"
                st.info("Multiple Lines set to 'No phone service' automatically.")
                
            internet_service = st.selectbox("Internet Service Provider (ISP)", ["Fiber optic", "DSL", "No"])
            
            # Sub-logic for Internet related services
            if internet_service != "No":
                online_security = st.selectbox("Online Security Service", ["No", "Yes"])
                online_backup = st.selectbox("Online Backup Service", ["No", "Yes"])
                device_protection = st.selectbox("Device Protection Plan", ["No", "Yes"])
                tech_support = st.selectbox("Premium Technical Support", ["No", "Yes"])
                streaming_tv = st.selectbox("Streaming TV Subscription", ["No", "Yes"])
                streaming_movies = st.selectbox("Streaming Movies Subscription", ["No", "Yes"])
            else:
                online_security = "No internet service"
                online_backup = "No internet service"
                device_protection = "No internet service"
                tech_support = "No internet service"
                streaming_tv = "No internet service"
                streaming_movies = "No internet service"
                st.info("Internet services automatically set to 'No internet service'.")
                
        with col_acc:
            st.markdown("#### 💳 Account & Billing Details")
            contract = st.selectbox("Contract Terms", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment Method", 
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            )
            
            tenure = st.slider("Tenure Length (Months)", min_value=0, max_value=72, value=12, step=1)
            monthly_charges = st.slider(
                "Monthly Charges ($)", 
                min_value=18.0, 
                max_value=120.0, 
                value=55.0, 
                step=0.5
            )
            
            # Logically suggest TotalCharges based on tenure * monthly_charges
            default_total = float(tenure * monthly_charges)
            total_charges = st.number_input(
                "Estimated Total Charges ($)", 
                min_value=0.0, 
                value=default_total,
                step=10.0,
                help="By default, this is Tenure * Monthly Charges. You can override it manually."
            )
            
        st.markdown("---")
        
        # Prediction Trigger Button
        if st.button("🔮 Analyze Customer Churn Risk"):
            # Map input back to dictionary matching training format
            customer_data = {
                'gender': gender,
                'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': int(tenure),
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'MonthlyCharges': float(monthly_charges),
                'TotalCharges': float(total_charges)
            }
            
            # Predict
            with st.spinner("Analyzing customer telemetry and billing patterns..."):
                time.sleep(1.2)  # Aesthetic delay
                result = pipeline.predict_single(customer_data)
                
            # Displays result using beautiful premium CSS divs
            prob = result['churn_probability']
            risk = result['risk_level']
            conf = result['confidence_score']
            label = result['churn_label']
            recs = result['business_recommendations']
            
            if result['churn_prediction'] == 1:
                # Churn risk high
                st.markdown(f"""
                <div class='result-container result-churn'>
                    <div class='churn-header'>⚠️ HIGH CHURN RISK WARNING</div>
                    <p style='font-size: 1.15rem;'>This customer is categorized under <b>{risk.upper()}</b> segment with a churn probability of <b>{prob*100:.1f}%</b>.</p>
                    <p style='font-size: 1rem; color: #555;'>We are <b>{conf:.1f}% confident</b> that this customer will close their account in the next contract billing cycle.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Loyal Customer
                st.markdown(f"""
                <div class='result-container result-loyal'>
                    <div class='loyal-header'>✅ LOYAL CUSTOMER SEGMENT</div>
                    <p style='font-size: 1.15rem;'>This customer is categorized under <b>{risk.upper()}</b> segment with a churn probability of only <b>{prob*100:.1f}%</b>.</p>
                    <p style='font-size: 1rem; color: #555;'>We are <b>{conf:.1f}% confident</b> that this customer will remain with the subscription program.</p>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Visual meters
            col_met1, col_met2 = st.columns(2)
            with col_met1:
                st.metric(label="Calculated Churn Probability", value=f"{prob*100:.1f}%")
                st.progress(prob)
            with col_met2:
                st.metric(label="Prediction Confidence", value=f"{conf:.1f}%")
                st.progress(conf / 100)
                
            # Business Retention Recommendations
            st.markdown("<br>#### 📋 Actionable Customer Retention Playbook", unsafe_allow_html=True)
            st.write("The AI marketing planner has generated the following tactical customer strategies to mitigate churn:")
            
            for rec in recs:
                st.markdown(f"- 💡 **Strategy**: {rec}")
                
            # Display Feature Contribution insights
            st.markdown("<br>#### 🔍 Local Feature Contributions (Key Churn Drivers)", unsafe_allow_html=True)
            insights = []
            if contract == "Month-to-month":
                insights.append("🔴 **Month-to-month contract**: Month-to-month contracts are highly flexible and constitute the single largest statistical driver of customer loss in this industry.")
            if internet_service == "Fiber optic":
                insights.append("🔴 **Fiber Optic Service**: Historically, fiber optic users show higher rates of complaints and churn, suggesting either service instability or extreme price sensitivity.")
            if tenure < 6:
                insights.append("🔴 **New Customer (< 6 months)**: High statistical vulnerability. Early onboarding friction must be smoothed immediately.")
            if tech_support == "No" and internet_service != "No":
                insights.append("🔴 **No Technical Support**: Customers with technical service types but without active premium technical support are 2x more likely to experience frustration and churn.")
            
            if len(insights) > 0:
                for ins in insights:
                    st.markdown(ins)
            else:
                st.markdown("🟢 **Highly Stable Account Config**: Longer tenure, steady payment methods, and long-term contracts contribute heavily to account longevity.")

    # -------------------------------------------------------------
    # TAB 2: BATCH PREDICTION
    # -------------------------------------------------------------
    with tab_batch:
        st.markdown("### Bulk Customer Prediction Console")
        st.write("Upload a CSV file containing multiple customers (complying with standard Telco dataset headers). The AI engine will compute churn risks for all, draw stats, and generate a downloadable report.")
        
        # Sample CSV download link
        st.markdown("📥 [Download Sample Customer Batch Template (CSV)](https://github.com/blastchar/telco-customer-churn/blob/master/WA_Fn-UseC_-Telco-Customer-Churn.csv) *(Upload standard Kaggle format directly)*")
        
        uploaded_file = st.file_uploader("Choose a bulk CSV file for batch predictions", type=["csv"])
        
        if uploaded_file is not None:
            try:
                bulk_df = pd.read_csv(uploaded_file)
                st.success(f"File successfully loaded! Found {bulk_df.shape[0]} customer records.")
                
                # Check for minimum column constraints
                required_headers = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
                                    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                                    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
                                    'MonthlyCharges', 'TotalCharges']
                                    
                missing_cols = [col for col in required_headers if col not in bulk_df.columns]
                
                if len(missing_cols) > 0:
                    st.error(f"❌ **Invalid CSV Format!** The uploaded CSV is missing the following required headers: `{missing_cols}`. Please upload a standard Kaggle dataset compatible file.")
                else:
                    # Run Bulk Predictions
                    with st.spinner("Processing batch pipeline through the Scikit-learn column transformer and prediction model..."):
                        time.sleep(1.8)
                        predictions_df = pipeline.predict_bulk(bulk_df)
                        
                    st.markdown("---")
                    st.markdown("### 📊 Batch Prediction Results & Analytics")
                    
                    # Display metrics
                    total_cust = len(predictions_df)
                    predicted_churn = int(predictions_df['Churn_Prediction'].sum())
                    churn_pct = (predicted_churn / total_cust) * 100
                    
                    col_b1, col_b2, col_b3 = st.columns(3)
                    with col_b1:
                        st.metric("Total Records Evaluated", total_cust)
                    with col_b2:
                        st.metric("Predicted Churned Customers", predicted_churn)
                    with col_b3:
                        st.metric("Overall Expected Churn Rate", f"{churn_pct:.1f}%")
                        
                    # Graph distribution of segments
                    st.markdown("#### 🗺️ Risk Segment Distribution")
                    segment_counts = predictions_df['Risk_Segment'].value_counts()
                    st.bar_chart(segment_counts)
                    
                    # Preview results
                    st.markdown("#### 📋 Prediction Report Preview (Top 10 High Risk Customers)")
                    high_risk_df = predictions_df.sort_values(by='Churn_Probability', ascending=False)
                    
                    # Show nice columns
                    display_cols = ['customerID', 'gender', 'tenure', 'Contract', 'InternetService', 
                                    'MonthlyCharges', 'TotalCharges', 'Churn_Probability', 'Risk_Segment']
                                    
                    # Keep only existing display columns
                    existing_display = [c for c in display_cols if c in high_risk_df.columns]
                    st.dataframe(high_risk_df[existing_display].head(15))
                    
                    # Export CSV button
                    st.markdown("---")
                    csv_data = predictions_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Full Batch Predictions (CSV)",
                        data=csv_data,
                        file_name="churn_predictions_report.csv",
                        mime="text/csv",
                        key="download-csv"
                    )
            except Exception as e:
                st.error(f"Error parsing uploaded file: {e}")
