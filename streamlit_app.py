import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Page config
st.set_page_config(
    page_title="Fetal Health Prediction App",
    page_icon="👶",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .normal {
        background-color: #C8E6C9;
        border: 2px solid #4CAF50;
    }
    .suspect {
        background-color: #FFECB3;
        border: 2px solid #FFC107;
    }
    .pathological {
        background-color: #FFCDD2;
        border: 2px solid #F44336;
    }
</style>
""", unsafe_allow_html=True)

# Load the model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('models/model.h5')
    scaler = joblib.load('models/scaler.h5')
    return model, scaler

try:
    model, scaler = load_model()
    model_load_success = True
except Exception as e:
    model_load_success = False
    st.error(f"Error loading model: {e}")

# App title and introduction
st.markdown('<h1 class="main-header">Fetal Health Prediction System</h1>', unsafe_allow_html=True)

with st.expander("ℹ️ About this app", expanded=False):
    st.markdown("""
    This application predicts fetal health based on cardiotocographic (CTG) examination data.
    
    **Features used for prediction:**
    - **Accelerations**: Number of accelerations per second
    - **Uterine Contractions**: Number of uterine contractions per second
    - **Prolonged Decelerations**: Number of prolonged decelerations per second
    - **Abnormal Short-term Variability**: Percentage of time with abnormal short-term variability
    - **Percentage of time with abnormal long-term variability**: Self-explanatory
    - **Mean value of long-term variability**: Self-explanatory
    - **Histogram features**: Mode, Mean, Median, and Variance of the histogram
    
    **Possible Outcomes:**
    - **Normal**: Indicates normal fetal health
    - **Suspect**: Indicates suspect fetal health, requiring closer monitoring
    - **Pathological**: Indicates pathological fetal health, requiring immediate medical attention
    
    This app is for educational purposes only and should not replace professional medical advice.
    """)

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Prediction", "Feature Information", "Sample Cases"])

with tab1:
    if model_load_success:
        st.markdown('<h2 class="sub-header">Enter Patient Data</h2>', unsafe_allow_html=True)
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            accelerations = st.number_input('Accelerations', min_value=0.0, max_value=1.0, value=0.0, step=0.01, help="Number of accelerations per second")
            uterine_contractions = st.number_input('Uterine Contractions', min_value=0.0, max_value=1.0, value=0.0, step=0.01, help="Number of uterine contractions per second")
            prolongued_decelerations = st.number_input('Prolonged Decelerations', min_value=0.0, max_value=1.0, value=0.0, step=0.01, help="Number of prolonged decelerations per second")
            abnormal_short_term_variability = st.number_input('Abnormal Short Term Variability', min_value=0.0, max_value=100.0, value=0.0, step=0.1, help="Percentage of time with abnormal short-term variability")
            percentage_of_time_with_abnormal_long_term_variability = st.number_input('% Time with Abnormal Long Term Variability', min_value=0.0, max_value=100.0, value=0.0, step=0.1)
        
        with col2:
            mean_value_of_long_term_variability = st.number_input('Mean Value of Long Term Variability', min_value=0.0, max_value=50.0, value=0.0, step=0.1)
            histogram_mode = st.number_input('Histogram Mode', min_value=0.0, max_value=500.0, value=0.0, step=1.0)
            histogram_mean = st.number_input('Histogram Mean', min_value=0.0, max_value=500.0, value=0.0, step=1.0)
            histogram_median = st.number_input('Histogram Median', min_value=0.0, max_value=500.0, value=0.0, step=1.0)
            histogram_variance = st.number_input('Histogram Variance', min_value=0.0, max_value=500.0, value=0.0, step=1.0)
        
        # Create prediction button with loading state
        with st.container():
            st.markdown("<br>", unsafe_allow_html=True)
            predict_col, reset_col = st.columns([3, 1])
            
            with predict_col:
                predict_button = st.button('📊 Predict Fetal Health', use_container_width=True)
            
            with reset_col:
                reset_button = st.button('🔄 Reset', use_container_width=True)
            
            if reset_button:
                st.rerun()
        
        # Make prediction
        if predict_button:
            with st.spinner('Processing data...'):
                # Create input data array
                inp_data = [
                    accelerations,
                    uterine_contractions,
                    prolongued_decelerations,
                    abnormal_short_term_variability,
                    percentage_of_time_with_abnormal_long_term_variability,
                    mean_value_of_long_term_variability,
                    histogram_mode,
                    histogram_mean,
                    histogram_median,
                    histogram_variance
                ]
                
                # Make prediction
                prediction = model.predict(scaler.transform([inp_data]))[0]
                
                # Get prediction probabilities
                prediction_proba = model.predict_proba(scaler.transform([inp_data]))[0]
                
                # Determine result class and display
                if prediction == 1:
                    fetal_health_type = 'Normal'
                    result_class = "normal"
                    icon = "✅"
                elif prediction == 2:
                    fetal_health_type = 'Suspect'
                    result_class = "suspect"
                    icon = "⚠️"
                elif prediction == 3:
                    fetal_health_type = 'Pathological'
                    result_class = "pathological"
                    icon = "🚨"
                
                # Display result
                st.markdown(f"<div class='result-box {result_class}'><h2>{icon} Predicted Fetal Health: {fetal_health_type}</h2></div>", unsafe_allow_html=True)
                
                # Create columns for visualization
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    # Display confidence levels
                    st.subheader("Prediction Confidence")
                    
                    # Create a DataFrame for the confidence levels
                    confidence_data = pd.DataFrame({
                        'Category': ['Normal', 'Suspect', 'Pathological'],
                        'Confidence': prediction_proba * 100
                    })
                    
                    # Create a bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = sns.barplot(x='Category', y='Confidence', data=confidence_data, palette=['#4CAF50', '#FFC107', '#F44336'], ax=ax)
                    
                    # Add percentage labels on top of bars
                    for i, bar in enumerate(bars.patches):
                        bars.text(bar.get_x() + bar.get_width()/2, 
                                bar.get_height() + 1, 
                                f"{prediction_proba[i]:.1%}", 
                                ha='center', fontsize=12, fontweight='bold')
                    
                    plt.title('Prediction Confidence Levels', fontsize=16)
                    plt.ylabel('Confidence (%)', fontsize=12)
                    plt.xlabel('', fontsize=12)
                    plt.ylim(0, 100)
                    st.pyplot(fig)
                
                with viz_col2:
                    # Display recommendations based on prediction
                    st.subheader("Recommendations")
                    
                    if prediction == 1:
                        st.success("""
                        ✅ **Normal fetal health detected**
                        
                        Recommendations:
                        - Continue regular prenatal check-ups
                        - Maintain healthy lifestyle and diet
                        - Monitor for any changes in fetal movement
                        - Follow standard prenatal care guidelines
                        """)
                    elif prediction == 2:
                        st.warning("""
                        ⚠️ **Suspect fetal health detected**
                        
                        Recommendations:
                        - Increased monitoring frequency
                        - Additional CTG examinations may be required
                        - Consider follow-up specialized evaluations
                        - Ensure adequate rest and hydration
                        - Consult healthcare provider promptly
                        """)
                    elif prediction == 3:
                        st.error("""
                        🚨 **Pathological fetal health detected**
                        
                        Recommendations:
                        - Immediate medical attention required
                        - Continuous fetal monitoring recommended
                        - Further diagnostic testing needed
                        - Prepare for possible medical intervention
                        - Consult with specialist immediately
                        """)
                    
                    st.info("⚠️ This is for educational purposes only. Always consult healthcare professionals for medical advice.")

with tab2:
    st.markdown('<h2 class="sub-header">Feature Information</h2>', unsafe_allow_html=True)
    
    # Feature descriptions and importance
    feature_info = pd.DataFrame({
        'Feature': [
            'Accelerations', 
            'Uterine Contractions', 
            'Prolonged Decelerations', 
            'Abnormal Short-term Variability',
            'Percentage of time with abnormal long-term variability',
            'Mean value of long-term variability',
            'Histogram Mode',
            'Histogram Mean',
            'Histogram Median',
            'Histogram Variance'
        ],
        'Description': [
            'Number of accelerations per second. Accelerations are temporary increases in fetal heart rate.',
            'Number of uterine contractions per second.',
            'Number of prolonged decelerations per second. Decelerations are temporary decreases in fetal heart rate.',
            'Percentage of time with abnormal short-term variability.',
            'Percentage of time with abnormal long-term variability.',
            'Mean value of long-term variability.',
            'Mode (most common value) of the histogram of fetal heart rate.',
            'Mean (average) of the histogram of fetal heart rate.',
            'Median (middle value) of the histogram of fetal heart rate.',
            'Variance (spread) of the histogram of fetal heart rate.'
        ],
        'Normal Range': [
            '0.0 - 0.15',
            '0.0 - 0.1',
            '0.0 - 0.01',
            '0.0 - 40.0',
            '0.0 - 30.0',
            '5.0 - 25.0',
            '120 - 160',
            '120 - 160',
            '120 - 160',
            '10 - 50'
        ]
    })
    
    st.dataframe(feature_info, use_container_width=True, hide_index=True)
    
    # CTG Information
    st.subheader("About Cardiotocography (CTG)")
    st.markdown("""
    Cardiotocography (CTG) is a technical means of recording the fetal heartbeat and the uterine contractions during pregnancy.
    The machine used to perform the monitoring is called a cardiotocograph, more commonly known as an electronic fetal monitor.
    
    The CTG reading is widely used during pregnancy to monitor fetal well-being and to evaluate fetal heart rate. 
    It is especially important during labor to detect signs of fetal distress.
    
    The interpretation of CTG recordings is crucial for clinical decision-making, including whether to deliver the baby surgically (by cesarean section).
    """)

with tab3:
    st.markdown('<h2 class="sub-header">Sample Cases</h2>', unsafe_allow_html=True)
    st.markdown("Use these sample cases to test the model quickly.")
    
    # Sample cases
    sample_cases = {
        "Normal Case": [0.008, 0.004, 0.0, 19.0, 0.0, 9.0, 132.0, 136.0, 138.0, 12.0],
        "Suspect Case": [0.002, 0.015, 0.002, 60.0, 30.0, 7.0, 128.0, 137.0, 140.0, 18.0],
        "Pathological Case": [0.0, 0.006, 0.008, 85.0, 70.0, 3.0, 133.0, 135.0, 130.0, 25.0]
    }
    
    # Create columns for sample cases
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Load Normal Case", use_container_width=True):
            st.session_state['sample_data'] = sample_cases["Normal Case"]
            st.rerun()
    
    with col2:
        if st.button("Load Suspect Case", use_container_width=True):
            st.session_state['sample_data'] = sample_cases["Suspect Case"]
            st.rerun()
    
    with col3:
        if st.button("Load Pathological Case", use_container_width=True):
            st.session_state['sample_data'] = sample_cases["Pathological Case"]
            st.rerun()
    
    # Use sample data if available in session state
    if 'sample_data' in st.session_state:
        # Display the loaded sample data
        sample_data = st.session_state['sample_data']
        st.success("Sample data loaded! Go to the Prediction tab and click 'Predict Fetal Health' to see results.")
        
        # Create DataFrame for display
        feature_names = [
            'Accelerations', 
            'Uterine Contractions', 
            'Prolonged Decelerations', 
            'Abnormal Short-term Variability',
            '% Time with Abnormal Long-term Variability',
            'Mean value of Long-term Variability',
            'Histogram Mode',
            'Histogram Mean',
            'Histogram Median',
            'Histogram Variance'
        ]
        
        sample_df = pd.DataFrame({
            'Feature': feature_names,
            'Value': sample_data
        })
        
        st.dataframe(sample_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("© 2025 Fetal Health Prediction App | Developed using Streamlit")
st.markdown("⚠️ Disclaimer: This application is for educational and demonstration purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.")
