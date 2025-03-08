import joblib
import streamlit as st

# Load the model and scaler
model = joblib.load('models/model.h5')
scaler = joblib.load('models/scaler.h5')

# Streamlit app
st.title('Fetal Health Classification')

st.write("Enter the following features:")

accelerations = st.text_input("Accelerations")
uterine_contractions = st.text_input("Uterine Contractions")
prolongued_decelerations = st.text_input("Prolongued Decelerations")
abnormal_short_term_variability = st.text_input("Abnormal Short Term Variability")
percentage_of_time_with_abnormal_long_term_variability = st.text_input("Percentage of Time with Abnormal Long Term Variability")
mean_value_of_long_term_variability = st.text_input("Mean Value of Long Term Variability")
histogram_mode = st.text_input("Histogram Mode")
histogram_mean = st.text_input("Histogram Mean")
histogram_median = st.text_input("Histogram Median")
histogram_variance = st.text_input("Histogram Variance")

if st.button('Predict'):
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
    
    fetal_health_type = model.predict(scaler.transform([inp_data]))[0]

    if fetal_health_type == 1:
        fetal_health_type = 'Normal'
    elif fetal_health_type == 2:
        fetal_health_type = 'Suspect'
    elif fetal_health_type == 3:
        fetal_health_type = 'Pathological'

    st.write(f"Fetal Health Type: {fetal_health_type}")
