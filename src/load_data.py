import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.store_data = None

    def load_data(self, train_path, test_path, store_path):
        """
        Load train, test, and store data from CSV files.
        
        Args:
            train_path (str): Path to the training data CSV file.
            test_path (str): Path to the test data CSV file.
            store_path (str): Path to the store data CSV file.
        """
        logging.info("Loading data...")
        try:
            self.train_data = pd.read_csv(train_path)
            self.test_data = pd.read_csv(test_path)
            self.store_data = pd.read_csv(store_path)
            logging.info("Data loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def merge_data(self):
        """
        Merge train and test data with store data.
        
        Returns:
            tuple: Merged train and test DataFrames.
        """
        logging.info("Merging data...")
        try:
            merged_train = pd.merge(self.train_data, self.store_data, on='Store', how='left')
            merged_test = pd.merge(self.test_data, self.store_data, on='Store', how='left')
            logging.info("Data merged successfully.")
            return merged_train, merged_test, self.store_data
        except Exception as e:
            logging.error(f"Error merging data: {str(e)}")
            raise

if __name__ == "__main__":
    loader = DataLoader()
    loader.load_data('../resources/Data/train.csv', '../resources/Data/test.csv', '../resources/Data/store.csv')
    merged_train, merged_test = loader.merge_data()
    print(merged_train.head())
    print(merged_test.head())