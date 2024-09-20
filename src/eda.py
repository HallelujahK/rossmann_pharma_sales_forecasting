import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EDAAnalyzer:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def analyze_promotions(self):
        """
        Analyze the distribution of promotions in training and test sets.
        """
        logging.info("Analyzing promotions distribution...")
        try:
            train_promo = self.train_data['Promo'].value_counts(normalize=True).sort_index()
            test_promo = self.test_data['Promo'].value_counts(normalize=True).sort_index()
            
            # Prepare data for plotting
            data = pd.DataFrame({
                'Train': train_promo,
                'Test': test_promo
            })
            
            plt.figure(figsize=(10, 6))
            data.plot(kind='bar', width=0.8)
            plt.title('Distribution of Promotions in Train and Test Sets')
            plt.xlabel('Promo (0: No Promotion, 1: Promotion)')
            plt.ylabel('Proportion')
            plt.legend(title='Dataset')
            plt.xticks(range(2), ['No Promo', 'Promo'], rotation=0)
            
            # Add value labels on the bars
            for i in range(len(data.index)):
                for j, value in enumerate(data.iloc[i]):
                    plt.text(i, value, f'{value:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig('../notebooks/figures/promo_distribution.png')
            plt.show()
            
            logging.info("Promotions analysis completed.")
        except Exception as e:
            logging.error(f"Error analyzing promotions: {str(e)}")
            raise
    def analyze_holiday_effects(self):
        """
        Analyze sales behavior before, during, and after holidays.
        """
        logging.info("Analyzing holiday effects...")
        try:
            self.train_data['Date'] = pd.to_datetime(self.train_data['Date'])
            
            # Define holidays with extended periods
            holidays = {
                'New Year': ('12-25', '01-07'),  # Extended from Dec 25 to Jan 7
                'Easter': ('03-22', '04-30'),    # Extended to cover potential Easter dates
                'Halloween': ('10-24', '11-01'),  # Week before Halloween and day after
                'Midsummer': ('06-14', '06-28'),  # Week before and few days after (includes Black Friday)
                'Christmas': ('12-15', '12-31')  # Extended from Dec 15 to Dec 31
            }
            
            # Create a new column for holiday periods
            self.train_data['Holiday'] = 'No Holiday'
            for holiday, (start, end) in holidays.items():
                for year in range(2013, 2016):
                    start_date = pd.to_datetime(f'{year}-{start}')
                    end_date = pd.to_datetime(f'{year}-{end}')
                    if holiday == 'Thanksgiving':
                        # Calculate Thanksgiving date (4th Thursday of November)
                        thanksgiving = pd.to_datetime(f'{year}-11-01') + pd.Timedelta(days=(3-pd.to_datetime(f'{year}-11-01').weekday()+7)%7)
                        start_date = thanksgiving - pd.Timedelta(days=7)
                        end_date = thanksgiving + pd.Timedelta(days=6)
                    mask = (self.train_data['Date'] >= start_date) & (self.train_data['Date'] <= end_date)
                    self.train_data.loc[mask, 'Holiday'] = holiday
            
            # Calculate average sales for each day
            daily_sales = self.train_data.groupby(['Date', 'Holiday'])['Sales'].mean().reset_index()
            
            # Plot
            plt.figure(figsize=(20, 10))
            
            # Plot daily sales
            sns.scatterplot(data=daily_sales[daily_sales['Holiday'] == 'No Holiday'], 
                            x='Date', y='Sales', color='gray', alpha=0.5, label='No Holiday', s=20)
            
            # Plot holiday sales with different colors
            holiday_colors = {
                'New Year': 'blue', 
                'Easter': 'green', 
                'Halloween': 'red',
                'Midsummer': 'orange',
                'Christmas': 'cyan'
            }
            for holiday in holidays.keys():
                sns.scatterplot(data=daily_sales[daily_sales['Holiday'] == holiday], 
                                x='Date', y='Sales', color=holiday_colors[holiday], label=holiday, s=30)
            
            plt.title('Average Daily Sales During Holiday Periods', fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Average Sales', fontsize=12)
            plt.legend(title='Holiday Period', title_fontsize='12', fontsize='10')
            
            # Highlight holiday periods
            for holiday, (start, end) in holidays.items():
                for year in range(2013, 2016):
                    if holiday == 'Thanksgiving':
                        thanksgiving = pd.to_datetime(f'{year}-11-01') + pd.Timedelta(days=(3-pd.to_datetime(f'{year}-11-01').weekday()+7)%7)
                        start_date = thanksgiving - pd.Timedelta(days=7)
                        end_date = thanksgiving + pd.Timedelta(days=6)
                    else:
                        start_date = pd.to_datetime(f'{year}-{start}')
                        end_date = pd.to_datetime(f'{year}-{end}')
                    plt.axvspan(start_date, end_date, alpha=0.2, color=holiday_colors[holiday])
            
            plt.tight_layout()
            plt.savefig('../notebooks/figures/holiday_sales_extended.png')
            plt.show()
            
            logging.info("Holiday effects analysis completed.")
        except Exception as e:
            logging.error(f"Error analyzing holiday effects: {str(e)}")
            raise

    def analyze_seasonality(self):
        """
        Identify seasonal purchasing behaviors.
        """
        logging.info("Analyzing seasonality...")
        try:
            self.train_data['Date'] = pd.to_datetime(self.train_data['Date'])
            self.train_data['Month'] = self.train_data['Date'].dt.month
            monthly_sales = self.train_data.groupby('Month')['Sales'].mean()
            
            plt.figure(figsize=(10, 6))
            monthly_sales.plot(kind='line', marker='o')
            plt.title('Average Monthly Sales')
            plt.xlabel('Month')
            plt.ylabel('Average Sales')
            plt.xticks(range(1, 13))
            plt.tight_layout()
            plt.savefig('../notebooks/figures/monthly_sales.png')
            plt.show()
            
            logging.info("Seasonality analysis completed.")
        except Exception as e:
            logging.error(f"Error analyzing seasonality: {str(e)}")
            raise

    def analyze_sales_customers_correlation(self):
        """
        Determine the correlation between sales and number of customers.
        """
        logging.info("Analyzing sales and customers correlation...")
        try:
            correlation = self.train_data['Sales'].corr(self.train_data['Customers'])
            
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='Customers', y='Sales', data=self.train_data.sample(n=1000))
            plt.title(f'Sales vs Customers (Correlation: {correlation:.2f})')
            plt.xlabel('Number of Customers')
            plt.ylabel('Sales')
            plt.tight_layout()
            plt.savefig('../notebooks/figures/sales_customers_correlation.png')
            plt.show()
            
            logging.info(f"Sales and customers correlation: {correlation:.2f}")
        except Exception as e:
            logging.error(f"Error analyzing sales and customers correlation: {str(e)}")
            raise

    def analyze_promotional_impact(self):
        """
        Investigate how promotions influence sales, customer numbers, and per-customer spending.
        """
        logging.info("Analyzing promotional impact...")
        try:
            # Group by Promo and calculate averages
            promo_impact = self.train_data.groupby('Promo').agg({
                'Sales': 'mean',
                'Customers': 'mean'
            }).reset_index()
            
            # Calculate average spend per customer
            promo_impact['Avg_Spend_Per_Customer'] = promo_impact['Sales'] / promo_impact['Customers']
            
            # Calculate percentage increases
            no_promo = promo_impact[promo_impact['Promo'] == 0].iloc[0]
            promo = promo_impact[promo_impact['Promo'] == 1].iloc[0]
            
            sales_increase = (promo['Sales'] - no_promo['Sales']) / no_promo['Sales'] * 100
            customer_increase = (promo['Customers'] - no_promo['Customers']) / no_promo['Customers'] * 100
            spend_per_customer_increase = (promo['Avg_Spend_Per_Customer'] - no_promo['Avg_Spend_Per_Customer']) / no_promo['Avg_Spend_Per_Customer'] * 100
            
            # Create plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Impact on Sales and Customers
            bar_width = 0.35
            index = np.arange(2)
            
            ax1.bar(index, promo_impact['Sales'], bar_width, label='Sales', color='b', alpha=0.5)
            ax1.bar(index + bar_width, promo_impact['Customers'], bar_width, label='Customers', color='r', alpha=0.5)
            
            ax1.set_xlabel('Promo')
            ax1.set_ylabel('Average Value')
            ax1.set_title('Impact of Promotions on Sales and Customers')
            ax1.set_xticks(index + bar_width / 2)
            ax1.set_xticklabels(['No Promo', 'Promo'])
            ax1.legend()
            
            # Plot 2: Impact on Average Spend per Customer
            ax2.bar(index, promo_impact['Avg_Spend_Per_Customer'], bar_width, color='g', alpha=0.5)
            
            ax2.set_xlabel('Promo')
            ax2.set_ylabel('Average Spend per Customer')
            ax2.set_title('Impact of Promotions on Average Spend per Customer')
            ax2.set_xticks(index)
            ax2.set_xticklabels(['No Promo', 'Promo'])
            
            plt.tight_layout()
            plt.savefig('../notebooks/figures/promo_impact_enhanced.png')
            plt.show()
            
            # Print analysis results
            print(f"Impact of Promotions:")
            print(f"Sales increase: {sales_increase:.2f}%")
            print(f"Customer increase: {customer_increase:.2f}%")
            print(f"Average spend per customer increase: {spend_per_customer_increase:.2f}%")
            
            logging.info("Promotional impact analysis completed.")
        except Exception as e:
            logging.error(f"Error analyzing promotional impact: {str(e)}")
            raise

    def run_eda(self):
        """
        Run all EDA analyses.
        """
        logging.info("Running full EDA...")
        self.analyze_promotions()
        self.analyze_holiday_effects()
        self.analyze_seasonality()
        self.analyze_sales_customers_correlation()
        self.analyze_promotional_impact()
        logging.info("EDA completed successfully.")

if __name__ == "__main__":
    # Example usage
    from load_data import DataLoader
    from clean_data import DataCleaner

    loader = DataLoader()
    loader.load_data('C:/Users/Lidya/Downloads/Week-4/rossmann_pharma_sales_forecasting/data/raw/train.csv', 
                     'C:/Users/Lidya/Downloads/Week-4/rossmann_pharma_sales_forecasting/data/raw/test.csv', 
                     'C:/Users/Lidya/Downloads/Week-4/rossmann_pharma_sales_forecasting/data/raw/store.csv')
    merged_train, merged_test = loader.merge_data()

    cleaner = DataCleaner()
    cleaned_train = cleaner.preprocess_data(merged_train)
    cleaned_test = cleaner.preprocess_data(merged_test)

    analyzer = EDAAnalyzer(cleaned_train, cleaned_test)
    analyzer.run_eda()