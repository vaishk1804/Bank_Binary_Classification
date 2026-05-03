# Bank Marketing Binary Classification

A machine learning project that predicts whether a customer will subscribe to a term deposit based on bank marketing campaign data.

This project demonstrates an end-to-end classification workflow, including preprocessing, feature handling, model training, class-imbalance awareness, and business-focused evaluation.

## Project Overview

The goal of this project was to build a binary classification model that predicts customer subscription behavior for a bank marketing campaign.

The target variable is:

    y

Where:

    1 = customer subscribed to the term deposit
    0 = customer did not subscribe

## Problem Statement

Banks often run outreach campaigns for financial products, but contacting every customer equally can be expensive and inefficient.

This project addresses that problem by building a predictive model that can help prioritize customers who are more likely to subscribe. The model can support more targeted marketing outreach, better campaign planning, and improved use of sales resources.

## Dataset

The project uses a large bank marketing dataset with separate train and test files.

    Train data: 750,000 rows × 18 columns
    Test data: 250,000 rows × 17 columns

The training data includes customer, campaign, and interaction-related features along with the target variable `y`.

## Repository Structure

    Bank_Binary_Classification/
    ├── Bank_Binary_preprocessing.ipynb
    ├── Bank_Classification_model.ipynb
    ├── train.csv
    ├── test.csv
    └── README.md

## Workflow

### 1. Data Loading and Exploration

- Loaded train and test datasets
- Reviewed feature types and target distribution
- Checked missing values and duplicate records
- Explored numerical and categorical feature behavior

### 2. Data Preprocessing

- Cleaned and prepared raw input data
- Encoded categorical variables
- Scaled or transformed numerical variables where needed
- Prepared training and validation splits
- Ensured train/test preprocessing consistency

### 3. Leakage and Imbalance Awareness

The project considered two common issues in bank marketing classification:

- **Target leakage:** Some campaign-related variables can accidentally reveal outcome information if not handled carefully.
- **Class imbalance:** Subscription outcomes are often imbalanced, so model performance should not be judged using accuracy alone.

### 4. Model Training

The project compared multiple classification models:

- Logistic Regression
- Random Forest
- XGBoost

These models provided a mix of interpretable baseline performance and stronger tree-based predictive performance.

### 5. Model Evaluation

Models were evaluated using classification metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

## Business Interpretation

For a marketing campaign, the model should not only predict correctly but also support campaign decisions.

Important trade-offs include:

- Higher recall helps identify more potential subscribers.
- Higher precision helps reduce wasted outreach.
- Threshold tuning can help balance conversion opportunity against campaign cost.
- A ranked probability output can help the bank prioritize outreach lists.

## Tech Stack

**Language:** Python  
**Environment:** Jupyter Notebook  
**Libraries:** pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn  
**Methods:** Binary classification, preprocessing, feature encoding, model comparison, evaluation metrics

## How to Run

Clone the repository:

    git clone <your-repo-link>
    cd Bank_Binary_Classification

Install dependencies:

    pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter

Launch Jupyter Notebook:

    jupyter notebook

Run the notebooks in order:

    1. Bank_Binary_preprocessing.ipynb
    2. Bank_Classification_model.ipynb

## What I Focused On

- Building an end-to-end binary classification workflow
- Preparing large train and test datasets for modeling
- Handling categorical and numerical features
- Comparing linear and tree-based models
- Thinking beyond accuracy by considering recall, precision, F1-score, and ROC-AUC
- Connecting model evaluation to marketing decision-making

## Highlights

- Built a machine learning pipeline on a 750,000-row training dataset
- Compared Logistic Regression, Random Forest, and XGBoost models
- Addressed practical classification concerns such as class imbalance and leakage awareness
- Framed model performance around business campaign trade-offs
- Demonstrated applied ML, data preprocessing, and business analytics skills

## Future Improvements

- Add final model comparison table to the README
- Add ROC and precision-recall curve images
- Add confusion matrix visualizations
- Add feature-importance plots
- Add threshold-tuning analysis for campaign targeting
- Add a reusable prediction script for new customer records
- Add `requirements.txt` for easier reproducibility

## Status

Completed machine learning classification project. Built to demonstrate applied predictive modeling, preprocessing, evaluation, and business-focused analytics for bank marketing campaign optimization.
