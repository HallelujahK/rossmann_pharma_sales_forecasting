import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataCleaner:
    def __init__(self):
        self.numerical_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')

    def handle_missing_values(self, df):
        """
        Handle missing values in the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with handled missing values.
        """
        logging.info("Handling missing values...")
        try:
            # Identify numerical and categorical columns
            numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
            categorical_columns = df.select_dtypes(include=['object']).columns

            # Impute missing values
            df[numerical_columns] = self.numerical_imputer.fit_transform(df[numerical_columns])
            df[categorical_columns] = self.categorical_imputer.fit_transform(df[categorical_columns])

            logging.info("Missing values handled successfully.")
            return df
        except Exception as e:
            logging.error(f"Error handling missing values: {str(e)}")
            raise

    def handle_outliers(self, df, columns, method='IQR'):
        """
        Handle outliers in specified columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            columns (list): List of columns to check for outliers.
            method (str): Method to use for outlier detection ('IQR' or 'zscore').
        
        Returns:
            pd.DataFrame: DataFrame with handled outliers.
        """
        logging.info(f"Handling outliers using {method} method...")
        try:
            for col in columns:
                if method == 'IQR':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df[col] = df[col].clip(lower_bound, upper_bound)
                elif method == 'zscore':
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    df[col] = df[col].mask(z_scores > 3, df[col].median())
                else:
                    raise ValueError("Invalid method. Use 'IQR' or 'zscore'.")

            logging.info("Outliers handled successfully.")
            return df
        except Exception as e:
            logging.error(f"Error handling outliers: {str(e)}")
            raise

    def preprocess_data(self, df):
        """
        Preprocess the data by handling missing values and outliers.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
        
        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        logging.info("Preprocessing data...")
        try:
            df = self.handle_missing_values(df)
            numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
            df = self.handle_outliers(df, numerical_columns, method='IQR')
            logging.info("Data preprocessing completed successfully.")
            return df
        except Exception as e:
            logging.error(f"Error preprocessing data: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    from load_data import DataLoader

    loader = DataLoader()
    loader.load_data('../resources/Data/train.csv', '../resources/Data/test.csv', '../resources/Data/store.csv')
    merged_train, merged_test = loader.merge_data()

    cleaner = DataCleaner()
    cleaned_train = cleaner.preprocess_data(merged_train)
    cleaned_test = cleaner.preprocess_data(merged_test)

    print(cleaned_train.head())
    print(cleaned_test.head())
