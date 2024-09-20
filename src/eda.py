import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def plot_sales_distribution(df: pd.DataFrame, save_path: str = None):
    """
    Plot the distribution of sales.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing sales data.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Sales'], bins=30, kde=True)
    plt.title("Distribution of Sales")
    plt.xlabel("Sales")
    plt.ylabel("Frequency")
    
    if save_path:
        plt.savefig(save_path)
        logging.info(f"Sales distribution plot saved to {save_path}")
    
    plt.show()

def plot_promo_sales_comparison(df: pd.DataFrame, save_path: str = None):
    """
    Compare sales between promo and non-promo days.
    
    Args:
        df (pd.DataFrame): Input DataFrame with sales and promo columns.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Promo', y='Sales', data=df)
    plt.title("Sales on Promo vs Non-Promo Days")
    plt.xlabel("Promo Day (1 = Yes, 0 = No)")
    plt.ylabel("Sales")
    
    if save_path:
        plt.savefig(save_path)
        logging.info(f"Promo sales comparison plot saved to {save_path}")
    
    plt.show()

def correlation_heatmap(df: pd.DataFrame, save_path: str = None):
    """
    Plot a correlation heatmap of the numerical features.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    plt.figure(figsize=(12, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Heatmap")
    
    if save_path:
        plt.savefig(save_path)
        logging.info(f"Correlation heatmap saved to {save_path}")
    
    plt.show()

def eda_summary(df: pd.DataFrame):
    """
    Print basic EDA summary: descriptive statistics and missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
    """
    logging.info("Generating descriptive statistics...")
    print(df.describe())

    logging.info("Checking for missing values...")
    print("\nMissing Values:\n", df.isnull().sum())
    
def run_eda(filepath: str):
    """
    Run the full EDA process: summary stats, plots, etc.
    
    Args:
        filepath (str): Path to the dataset.
    """
    df = pd.read_csv(filepath)
    eda_summary(df)
    plot_sales_distribution(df)
    plot_promo_sales_comparison(df)
    correlation_heatmap(df)

    logging.info("EDA process completed")
