import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_distribution(data, column, title, xlabel, ylabel, save_path):
    """
    Plot the distribution of a column in the dataset.

    Args:
        data (pd.DataFrame): Input DataFrame.
        column (str): Column name to plot.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Distribution plot saved to {save_path}")

def plot_correlation_heatmap(data, title, save_path):
    """
    Plot a correlation heatmap for numerical columns in the dataset.

    Args:
        data (pd.DataFrame): Input DataFrame.
        title (str): Plot title.
        save_path (str): Path to save the plot.
    """
    corr = data.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Correlation heatmap saved to {save_path}")

def plot_time_series(data, date_column, value_column, title, xlabel, ylabel, save_path):
    """
    Plot a time series of a specific column.

    Args:
        data (pd.DataFrame): Input DataFrame.
        date_column (str): Name of the date column.
        value_column (str): Name of the value column to plot.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=date_column, y=value_column, data=data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Time series plot saved to {save_path}")

def calculate_summary_statistics(data, column):
    """
    Calculate summary statistics for a specific column.

    Args:
        data (pd.DataFrame): Input DataFrame.
        column (str): Column name to calculate statistics for.

    Returns:
        dict: Dictionary containing summary statistics.
    """
    summary = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max(),
        'q1': data[column].quantile(0.25),
        'q3': data[column].quantile(0.75)
    }
    return summary

if __name__ == "__main__":
    # Example usage
    from load_data import DataLoader
    from clean_data import DataCleaner

    loader = DataLoader()
    loader.load_data('../data/raw/train.csv', '../data/raw/test.csv', '../data/raw/store.csv')
    merged_train, merged_test = loader.merge_data()

    cleaner = DataCleaner()
    cleaned_train = cleaner.preprocess_data(merged_train)

    plot_distribution(cleaned_train, 'Sales', 'Distribution of Sales', 'Sales', 'Frequency', '../notebooks/sales_distribution.png')
    plot_correlation_heatmap(cleaned_train, 'Correlation Heatmap', '../notebooks/correlation_heatmap.png')
    plot_time_series(cleaned_train, 'Date', 'Sales', 'Sales Over Time', 'Date', 'Sales', '../notebooks/sales_time_series.png')

    sales_summary = calculate_summary_statistics(cleaned_train, 'Sales')
    print("Sales Summary Statistics:", sales_summary)