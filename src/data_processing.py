import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.
    
    Args:
        filepath (str): Path to the dataset file.
    
    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Data loaded successfully from {filepath}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found at {filepath}")
        return pd.DataFrame()

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values by filling or dropping them.
    
    Args:
        df (pd.DataFrame): DataFrame with missing values.
    
    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    # Fill missing numerical values with the median
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    # Fill missing categorical values with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    
    logging.info("Missing values handled")
    return df

def detect_outliers(df: pd.DataFrame, z_thresh: float = 3) -> pd.DataFrame:
    """
    Detect outliers using the Z-score method.
    
    Args:
        df (pd.DataFrame): DataFrame with potential outliers.
        z_thresh (float): Z-score threshold to identify outliers.
    
    Returns:
        pd.DataFrame: DataFrame with outliers flagged.
    """
    z_scores = np.abs((df - df.mean()) / df.std())
    outliers = (z_scores > z_thresh).sum()
    
    logging.info(f"Outliers detected in {outliers.sum()} rows")
    return df[z_scores < z_thresh].dropna()

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing columns.
    
    Args:
        df (pd.DataFrame): Original dataframe.
    
    Returns:
        pd.DataFrame: DataFrame with new features added.
    """
    # Example: Extract day, month, and year from a date column
    if 'Date' in df.columns:
        df['Year'] = pd.to_datetime(df['Date']).dt.year
        df['Month'] = pd.to_datetime(df['Date']).dt.month
        df['Day'] = pd.to_datetime(df['Date']).dt.day
    
    logging.info("Feature engineering complete")
    return df

def preprocess_data(filepath: str) -> pd.DataFrame:
    """
    Full preprocessing pipeline: load data, handle missing values, detect outliers, and feature engineering.
    
    Args:
        filepath (str): Path to the dataset.
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df = load_data(filepath)
    df = handle_missing_values(df)
    df = detect_outliers(df)
    df = feature_engineering(df)
    
    logging.info("Data preprocessing completed")
    return df