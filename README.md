# Fetal Health Classification

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-brightgreen)](https://share.streamlit.io/gamal-abdelhakm/Fetal-Health-Classification/main/streamlit_app.py)
 
This repository implements a fetal health classification system achieving 95.83% accuracy using a tuned Gradient Boosting Classifier.

## Overview

This project aims to predict fetal health based on cardiotocographic (CTG) examination data. The model can classify fetal health into three categories: Normal, Suspect, and Pathological.

## Features

- **Accelerations**: Number of accelerations per second
- **Uterine Contractions**: Number of uterine contractions per second
- **Prolonged Decelerations**: Number of prolonged decelerations per second
- **Abnormal Short-term Variability**: Percentage of time with abnormal short-term variability
- **Percentage of Time with Abnormal Long-term Variability**: Self-explanatory
- **Mean Value of Long-term Variability**: Self-explanatory
- **Histogram Features**: Mode, Mean, Median, and Variance of the histogram

## Model

The model was trained using a Gradient Boosting Classifier and tuned to achieve high accuracy. The final model and scaler are saved and loaded in the Streamlit application.

## Streamlit App

You can interact with the model using the Streamlit app. The app provides an intuitive interface to enter patient data and predict fetal health.

### How to Run the App

1. **Clone the repository**:
   ```bash
   git clone https://github.com/gamal-abdelhakm/Fetal-Health-Classification.git
   cd Fetal-Health-Classification
   ```

2. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run streamlit_app.py
   ```

## Usage

1. **Enter Patient Data**: Input the features related to fetal health.
2. **Predict**: Click the "Predict Fetal Health" button to get the prediction.
3. **View Results**: The app will display the predicted fetal health category along with confidence levels and recommendations.

## Example

The app provides sample cases to quickly test the model.

- **Normal Case**:
  - Accelerations: 0.008
  - Uterine Contractions: 0.004
  - Prolonged Decelerations: 0.0
  - Abnormal Short-term Variability: 19.0
  - Percentage of Time with Abnormal Long-term Variability: 0.0
  - Mean Value of Long-term Variability: 9.0
  - Histogram Mode: 132.0
  - Histogram Mean: 136.0
  - Histogram Median: 138.0
  - Histogram Variance: 12.0

- **Suspect Case**:
  - Accelerations: 0.002
  - Uterine Contractions: 0.015
  - Prolonged Decelerations: 0.002
  - Abnormal Short-term Variability: 60.0
  - Percentage of Time with Abnormal Long-term Variability: 30.0
  - Mean Value of Long-term Variability: 7.0
  - Histogram Mode: 128.0
  - Histogram Mean: 137.0
  - Histogram Median: 140.0
  - Histogram Variance: 18.0

- **Pathological Case**:
  - Accelerations: 0.0
  - Uterine Contractions: 0.006
  - Prolonged Decelerations: 0.008
  - Abnormal Short-term Variability: 85.0
  - Percentage of Time with Abnormal Long-term Variability: 70.0
  - Mean Value of Long-term Variability: 3.0
  - Histogram Mode: 133.0
  - Histogram Mean: 135.0
  - Histogram Median: 130.0
  - Histogram Variance: 25.0

## Disclaimer

This application is for educational and demonstration purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.

## Summary Video
A video summarizing what I did in the code can be found using the following link: [Summary Video](https://drive.google.com/file/d/1NsVqzhoBCpzv1NTg0eIF2nKc0npjz7tx/view)
