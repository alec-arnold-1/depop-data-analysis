import pandas as pd
from pathlib import Path
import re

def load_and_combine_data():
    """Finds all CSVs in the /data folder and merges them"""
    
    data_dir = Path.cwd() / "data"

    # get list of all CSV files
    csv_files = list(data_dir.glob("*.csv"))

    if not csv_files:
        print(f"Checking in: {data_dir}")
        raise FileNotFoundError("No CSV files found in the /data folder!")

    print(f"Found {len(csv_files)} files: {[f.name for f in csv_files]}")

    # read and combine
    df_list = []
    for file in csv_files:
        temp_df = pd.read_csv(file)
        # track which file data is from
        temp_df['source_file'] = file.name
        df_list.append(temp_df)
        print(f"Successfully loaded {file.name} with {len(temp_df)} rows.")

    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"\nTotal rows in combined dataset: {len(combined_df)}")

    return combined_df

def clean_data(df):
    """Cleaning function for dataframe to be used in EDA and feature engineering"""
    # standardize column names
    df.columns = df.columns.str.replace(" ", "_").str.replace("-", "_").str.lower()

    # drop irrelevant columns
    buyer_info = ['buyer', 'name', 'address_line_1', 'address_line_2', 'city', 'post_code', 'country','state']
    irrelevant = [
        'fees_refunded_to_seller', 'refunded_to_buyer_amount', 'us_sales_tax',
        'payout_arrival_date', 'payment_type', 'estimated_payout_date',
        'payout_id', 'source_file', 'usps_cost', 'depop_payments_fee', 'buyer_marketplace_fee'
    ]
    df = df.drop(columns=buyer_info + irrelevant, errors='ignore')

    # convert currency strings to floats
    currency_cols = ['item_price', 'buyer_shipping_cost', 'total', 'depop_fee', 'boosting_fee']
    for col in currency_cols:
        if col in df.columns:
            # currency cleaning, values/decimals only
            df[col] = df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # standardize date and time
    df['date_of_listing'] = pd.to_datetime(df['date_of_listing'], errors='coerce')
    df['date_of_sale'] = pd.to_datetime(df['date_of_sale'], errors='coerce')

    # drop row if essential columns are null
    df = df.dropna(subset=['date_of_listing', 'date_of_sale', 'item_price'])

    # only keep rows where sale happens after or on listing date
    df = df[df['date_of_sale'] >= df['date_of_listing']]

    # drop duplicate rows
    df = df.drop_duplicates()

    return df

def create_features(df):
    """Create the features needed for analysis and ML"""
    df = df.copy()
    # Time-based features
    df['days_to_sell'] = (df['date_of_sale'] - df['date_of_listing']).dt.days
    df['day_posted'] = df['date_of_listing'].dt.day_name()
    
    # Combine Date and Time strings
    df['full_sale_timestamp'] = df['date_of_sale'].dt.strftime('%Y-%m-%d') + ' ' + df['time_of_sale']

    # Convert to datetime
    df['full_sale_timestamp'] = pd.to_datetime(df['full_sale_timestamp'], errors='coerce')

    # Extract the actual hour
    df['hour_sold'] = df['full_sale_timestamp'].dt.hour

    # Feature: is it boosted? (binary)
    df['is_boosted'] = df['boosting_fee'] > 0

    # Description based features
    df['hashtags'] = df['description'].apply(lambda x: re.findall(r"#\w+", str(x)))
    df['hashtag_count'] = df['hashtags'].apply(len)
    df['description_word_count'] = df['description'].apply(lambda x: len(str(x).split()))

    # Define "Fast Sale" (target variable, true if sold in < 10 days)
    df['fast_sale'] = df['days_to_sell'] < 10

    # Feature: Price relative to the average for that category
    df['category_avg_price'] = df.groupby('category')['item_price'].transform('mean')
    df['price_ratio'] = df['item_price'] / df['category_avg_price']

    # Feature: Seasonality
    df['month_listed'] = df['date_of_listing'].dt.month
    # Grouping months into seasons (Winter, Spring, Summer, Fall)
    # 12, 1, 2 = Winter | 3, 4, 5 = Spring | 6, 7, 8 = Summer | 9, 10, 11 = Fall
    df['season'] = df['month_listed'].apply(lambda x:
        'Winter' if x in [12, 1, 2] else
        'Spring' if x in [3, 4, 5] else
        'Summer' if x in [6, 7, 8] else 'Fall')

    return df

def handle_cardinality(df, column, top_n=20):
    # standardize to Title Case (converts 'other' -> 'Other', 'NIKE' -> 'Nike')
    df[column] = df[column].astype(str).str.title().str.strip()

    # Get the top N most frequent values
    top_values = df[column].value_counts().nlargest(top_n).index

    # If it's not in the top N, rename it to 'Other'
    # merges engineered 'other' and Depop's 'Other'
    df[column] = df[column].apply(lambda x: x if x in top_values else 'Other')

    return df

def prepare_ml_dataframe(df, predictors, categoricals, targets):
    """create dataframe for ML: encodes categories and filters features."""

    # One-hot encoding for categorical variables
    df_encoded = pd.get_dummies(df, columns=categoricals, drop_first=True)

    
    # grab new dummy columns
    dummy_cols = [c for c in df_encoded.columns if any(cat + "_" in c for cat in categoricals)]

    # combine base numeric features with the new dummy features
    all_features = predictors + dummy_cols

    # filter the dataframe to only include our chosen inputs and targets
    # keeps out any interaction/leakage columns
    ml_df = df_encoded[all_features + targets].copy()

    # final data integrity check
    # convert everything to numeric (Booleans become 1/0) and drop any NaNs
    ml_df = ml_df.apply(pd.to_numeric, errors='coerce').dropna()

    return ml_df, all_features