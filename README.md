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
### Installation

```bash
pip install -r requirements.txt
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

## Technical Challenges

**Small Sample Size (N = 301)**: With only 301 total observations, the model is prone to high variance.

**Class Imbalance & Median Thresholding**: Initial target variable (fast sale being less than 10 days) had a heavy bias towards fast sales (72%). I redefined the target variable based on the statistical median (3 days) in order to provide a more honest assessment of listing performance.

**Lack of Features**: Depop as a marketplace relies heavily on other features not found in the exports sales data CSVs such as quality of photos, frequency of the seller creating listings, and review rating. This likely accounts for the ceiling of predictive accuracy.

**Selection Bias**: The dataset only contains successful transactions (tems that ended up selling). It does not ionclude active listings that have failed to sell. The model is trained to distinguish between fast and slow sales, but cannot predict no sales. The model potentially ignores the features that lead to a total lack of purchase.

## Future Steps

**Computer Vision Integration**: Extract features like brightness, background consistency, etc to account for the visual aspect of Depop listings.

**Deployment as a Web App**: Wrap the pipeline into a FastAPI backend with a React frontend, allowing users to import and automatically see visualizations of their sales data. Also implement feature to enter a draft listing to receive a prediction on sale velocity based on factors like price.
