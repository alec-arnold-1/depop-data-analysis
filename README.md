# Depop Sales Analysis & Fast Sale Prediction

A comprehensive machine learning pipeline for analyzing Depop marketplace sales data and predicting fast sales. This project demonstrates end-to-end data science skills from data cleaning to predictive modeling.

## Project Overview

This project analyzes Depop sales data to understand what factors drive fast sales and builds a predictive model to identify items likely to sell quickly. The pipeline processes raw sales data, engineers meaningful features, and trains an XGBoost classifier to predict fast sales (items sold in less than 3 days).

## Key Features

- **Data Pipeline**: Automated loading, cleaning, and feature engineering
- **Temporal Analysis**: Time-based patterns and seasonality insights
- **Feature Engineering**: Custom features like price ratios, hashtag analysis, and boosted item detection
- **Predictive Modeling**: XGBoost classifier with comprehensive evaluation metrics
- **Professional EDA**: Detailed visualizations and statistical analysis

## Project Structure

```
ML_Pipeline_EDA/
├── pipeline_utils.py          # Core data processing and ML pipeline
├── exploratory_analysis.ipynb # Comprehensive EDA and visualizations
├── predictive_modeling.ipynb  # XGBoost model implementation
├── data/                      # Raw depop sales data (CSV files)
└── README.md                  # This file
```

## Technical Skills Demonstrated

- **Data Processing**: Pandas, data cleaning, feature engineering
- **Machine Learning**: XGBoost, model evaluation, hyperparameter tuning
- **Data Visualization**: Matplotlib, Seaborn, comprehensive EDA
- **Software Engineering**: Modular code design, reusable functions
- **Statistical Analysis**: Correlation analysis, outlier detection
- **Time Series Analysis**: Temporal patterns and seasonality

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost

### Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### Running the Project

1. **Data Preparation**: Place your Depop sales CSV files in the `data/` directory
2. **Exploratory Analysis**: Run `exploratory_analysis.ipynb` for comprehensive EDA
3. **Predictive Modeling**: Run `predictive_modeling.ipynb` for model training and evaluation

## Data Pipeline

The project follows a structured data pipeline:

1. **Data Loading**: Automatically combines all CSV files from the data directory
2. **Data Cleaning**: Standardizes columns, handles missing values, removes duplicates
3. **Feature Engineering**: Creates time-based features, text analysis features, and price ratios
4. **Cardinality Handling**: Groups rare categories into "Other" for better model performance
5. **ML Preparation**: One-hot encodes categorical variables and prepares final dataset

## Key Insights & Findings

- **Seasonal Patterns**: Sales vary significantly by season and day of week
- **Price Impact**: Item price correlates with sales speed
- **Shipping Influence**: Shipping cost strongly correlates with sale speed
- **Timing Matters**: Specific hours and days yield better sales performance
- **Feature Importance**: Price ratio and boosted items are top predictors

## Business Value

This analysis provides actionable insights for Depop sellers:
- Optimal listing times and pricing strategies
- Understanding seasonal demand patterns
- Identifying items likely to sell quickly
- Data-driven decision making for inventory management

## Technologies Used

- **Python**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib/Seaborn**: Data visualization
- **Scikit-learn**: Machine learning algorithms and evaluation
- **XGBoost**: Gradient boosting classifier

