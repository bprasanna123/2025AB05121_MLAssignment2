# Bank Customer Churn Prediction – Machine Learning Assignment

## Problem Statement

Customer churn prediction is an important business problem where organizations aim to identify customers who are likely to leave their services. This project focuses on predicting bank customer churn using multiple machine learning classification models and deploying the solution through an interactive Streamlit web application.

---

## Dataset Description

The dataset used is the **Bank Customer Churn Prediction Dataset** obtained from Kaggle.  
It contains approximately **10,000 customer records** with demographic, financial, and behavioral attributes.

### Target Variable
- **Exited** → Indicates whether the customer left the bank (1) or stayed (0).

### Key Features
- Credit score, age, tenure, account balance, estimated salary  
- Geography and gender information  
- Banking activity indicators  

The dataset satisfies assignment requirements with more than **500 instances and over 12 features**.

---

## Data Preprocessing

The dataset undergoes preprocessing including:

- Removal of irrelevant identifier columns (`RowNumber`, `CustomerId`, `Surname`)
- Encoding categorical variables using one-hot encoding
- Feature scaling using StandardScaler
- Train-test split for unbiased evaluation

---

## Machine Learning Models Used

The following classification models are trained and evaluated:

- Logistic Regression  
- Decision Tree  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- Random Forest (Ensemble)  
- XGBoost (Ensemble)

---

## Model Comparison Table (Evaluation Metrics)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|------|----------|--------|------|------|
| Logistic Regression | 0.8080 | 0.7748 | 0.5891 | 0.1867 | 0.2836 | 0.2515 |
| Decision Tree | 0.7825 | 0.6815 | 0.4685 | 0.5111 | 0.4888 | 0.3516 |
| KNN | 0.8240 | 0.7531 | 0.6222 | 0.3440 | 0.4430 | 0.3703 |
| Naive Bayes | 0.8200 | 0.7843 | 0.6000 | 0.3464 | 0.4393 | 0.3594 |
| Random Forest (Ensemble) | 0.8640 | 0.8522 | 0.7824 | 0.4595 | 0.5789 | 0.5297 |
| XGBoost (Ensemble) | 0.8490 | 0.8328 | 0.6829 | 0.4816 | 0.5648 | 0.4874 |

---

## Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|--------------|------------------------------------|
| Logistic Regression | Shows reasonable accuracy but very low recall, indicating difficulty detecting churn customers. |
| Decision Tree | Provides moderate performance but may overfit compared to ensemble models. |
| KNN | Offers stable baseline performance but less effective than ensemble approaches. |
| Naive Bayes | Balanced probabilistic model but comparatively lower predictive power. |
| Random Forest (Ensemble) | Achieves the best overall performance with highest accuracy, AUC, F1 score, and MCC, indicating strong generalization capability. |
| XGBoost (Ensemble) | Performs strongly with competitive accuracy and recall, slightly below Random Forest. |

---

## Streamlit Web Application

An interactive Streamlit application demonstrates model predictions with the following features:

- CSV dataset upload option  
- Model selection dropdown  
- Display of evaluation metrics  
- Confusion matrix and classification report visualization  

### Deployment Link
https://bank-churn-ml-assignment-ffaltdqlezdcpzxvelbmks.streamlit.app/

---

## GitHub Repository

Repository Link:  
https://github.com/sthippa/bank-churn-ml-assignment

---

## Repository Structure

```
project-folder/
│-- app.py                    # Streamlit web application
│-- requirements.txt          # Python dependencies for deployment
│-- README.md                 # Project documentation
│-- model_training.ipynb      # Model training notebook
│-- Churn_Modelling.csv       # Dataset used for training
│-- model_metrics.csv         # Evaluation metrics of models
│-- scaler.pkl                # Saved scaler for preprocessing
│
├── model/
│   ├── logistic.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl

```


---

## Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Streamlit  
- Matplotlib & Seaborn  

---

## Assignment Execution Environment

This assignment is executed on the **BITS Virtual Lab environment** as required.

---

## Conclusion

This project demonstrates a complete machine learning pipeline including data preprocessing, model training, evaluation, deployment, and interactive visualization using Streamlit for customer churn prediction.