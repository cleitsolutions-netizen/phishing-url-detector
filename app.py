import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# Set Page Config
st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🛡️",
    layout="wide"
)

# Title & Description
st.title("🛡️ Malicious & Phishing URL Detection System")
st.markdown("""
This application uses an advanced **Random Forest Classifier** trained on 87 structural, content, and network features to determine whether a given web link is **Legitimate** or a **Phishing scam**.
""")

# Load Model Freely (Handles both raw model files or dictionary objects)
@st.cache_resource
def load_model_data():
    loaded_data = joblib.load('phishing_detector_model.pkl')
    if isinstance(loaded_data, dict):
        return loaded_data['model'], loaded_data['features']
    return loaded_data, list(loaded_data.feature_names_in_)

try:
    model, feature_names = load_model_data()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load Examples
@st.cache_data
def load_examples():
    with open('examples.json', 'r') as f:
        return json.load(f)

examples = load_examples()

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["🔍 Interactive URL Tester", "📊 Batch CSV Classifier"])

with tab1:
    st.header("Single URL Live Prediction Simulation")
    st.write("Select one of the benchmark URLs from our validation set to see the trained ML model extract features and compute safety metrics:")
    
    # Dropdown selections
    url_options = [ex['url'] for ex in examples]
    selected_url = st.selectbox("Choose a URL to analyze:", url_options)
    
    # Get the corresponding sample data
    sample_data = next(ex for ex in examples if ex['url'] == selected_url)
    features_dict = sample_data['features']
    true_label = sample_data['status']
    
    st.info(f"**Selected URL:** `{selected_url}`")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Key URL Features Found")
        display_features = {
            "URL Length": features_dict.get('length_url'),
            "Hostname Length": features_dict.get('length_hostname'),
            "Has IP Address?": "Yes" if features_dict.get('ip') == 1 else "No",
            "Number of Dots": features_dict.get('nb_dots'),
            "Google Index Status": "Indexed" if features_dict.get('google_index') == 1 else "Not Indexed",
            "PageRank Score": features_dict.get('page_rank'),
            "Web Traffic Ranking": features_dict.get('web_traffic'),
            "Suspicious TLD?": "Yes" if features_dict.get('suspecious_tld') == 1 else "No"
        }
        st.json(display_features)
        
    with col2:
        st.subheader("🤖 Machine Learning Prediction")
        
        # Prepare input for prediction (ensure exact column order)
        input_df = pd.DataFrame([features_dict])[feature_names]
        
        # Run prediction
        pred_code = model.predict(input_df)[0]
        pred_prob = model.predict_proba(input_df)[0]
        
        if pred_code == 1:
            st.error(f"🚨 **PREDICTION: PHISHING SITE**")
            st.metric(label="Phishing Confidence", value=f"{pred_prob[1]*100:.2f}%")
            st.markdown("⚠️ **Warning:** This URL exhibits structural traits common to credential harvesting pages, such as being non-indexed or lacking credible domain authority.")
        else:
            st.success(f"✅ **PREDICTION: LEGITIMATE SITE**")
            st.metric(label="Legitimate Confidence", value=f"{pred_prob[0]*100:.2f}%")
            st.markdown("🎉 **Safe:** This URL demonstrates patterns consistent with verified internet infrastructure, carrying established search reputation rankings.")

        st.caption(f"Ground Truth Dataset Label: **{true_label.upper()}**")

with tab2:
    st.header("Bulk Analysis Tool")
    st.write("Upload a CSV file with URL features to run high-throughput automated classification.")
    
    uploaded_file = st.file_uploader("Upload feature CSV dataset:", type=["csv"])
    
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(input_data)} rows successfully.")
        
        # Check if features match
        missing_feats = [f for f in feature_names if f not in input_data.columns]
        
        if missing_feats:
            st.error(f"Uploaded CSV is missing required model features.")
        else:
            if st.button("Run Batch Prediction"):
                with st.spinner("Processing rows..."):
                    preds = model.predict(input_data[feature_names])
                    probs = model.predict_proba(input_data[feature_names])
                    
                    results = input_data.copy()
                    results['Predicted_Status'] = np.where(preds == 1, 'phishing', 'legitimate')
                    results['Confidence_Score'] = np.max(probs, axis=1)
                    
                    st.success("Batch processing complete!")
                    st.dataframe(results[['url', 'Predicted_Status', 'Confidence_Score']].head(20) if 'url' in results.columns else results['Predicted_Status'].head(20))
                    
                    # Download button
                    csv_res = results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Complete Prediction Report CSV",
                        data=csv_res,
                        file_name="phishing_predictions_output.csv",
                        mime="text/csv"
                    )
