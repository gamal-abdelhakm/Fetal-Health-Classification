# Fetal-Health-Classification

The Fetal Health Prediction model is a machine learning model that uses various techniques to classify the health of a fetus and is deployed as a web application using the Streamlit framework. This model can be used by medical professionals to identify potential health issues in the fetus early on and take appropriate action to prevent child and maternal mortality. It can also be used to monitor the health of the fetus throughout pregnancy and provide personalized care to the expectant mother.   

Overview
This project focuses on classifying fetal health status using various machine learning algorithms. The goal is to predict the health status of a fetus based on a set of medical features. The dataset is evaluated using different classification models to identify the most accurate and reliable model.

Dataset
The dataset contains medical data for fetal health classification. The target variable represents the health status of the fetus, which can be one of three classes:
[Normal-Suspect-Pathological]

Models Evaluated:

Logistic Regression
Training Accuracy: 89.82%
Testing Accuracy: 88.80%
Macro Avg F1-Score: 0.78
Weighted Avg F1-Score: 0.89

Support Vector Classifier (SVC)
Training Accuracy: 91.90%
Testing Accuracy: 90.32%
Macro Avg F1-Score: 0.82
Weighted Avg F1-Score: 0.90

K-Nearest Neighbors (KNN)
Training Accuracy: 94.62%
Testing Accuracy: 91.65%
Macro Avg F1-Score: 0.84
Weighted Avg F1-Score: 0.91

Decision Tree Classifier
Training Accuracy: 99.94%
Testing Accuracy: 92.41%
Macro Avg F1-Score: 0.86
Weighted Avg F1-Score: 0.92

Random Forest Classifier
Training Accuracy: 99.94%
Testing Accuracy: 94.69%
Macro Avg F1-Score: 0.90
Weighted Avg F1-Score: 0.94

Gradient Boosting Classifier
Training Accuracy: 98.55%
Testing Accuracy: 95.64%
Macro Avg F1-Score: 0.92
Weighted Avg F1-Score: 0.95

Improved Gradient Boosting Classifier (Hyperparameter Tuning)
Training Accuracy: 99.94%
Testing Accuracy: 95.83%
Macro Avg F1-Score: 0.92
Weighted Avg F1-Score: 0.96

Key Findings: 
The Gradient Boosting Classifier with hyperparameter tuning achieved the highest testing accuracy and robust performance across different metrics.
The models demonstrate strong predictive capabilities, with the improved Gradient Boosting Classifier showing an impressive testing accuracy of 95.83%.

Conclusion: 
Machine learning models, particularly the Gradient Boosting Classifier with tuned hyperparameters, are effective in classifying fetal health statuses based on medical data. These models can assist healthcare professionals in making informed decisions about fetal health.

A video summarizing what I did in the code : https://drive.google.com/file/d/1NsVqzhoBCpzv1NTg0eIF2nKc0npjz7tx/view
