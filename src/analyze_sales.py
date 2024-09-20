import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SalesAnalyzer:
    def __init__(self, data):
        self.data = data

    def analyze_promo_effectiveness(self):
        """
        Analyze if promotions could be deployed more effectively.
        Identify stores where promotions would yield the highest impact.
        """
        logging.info("Analyzing promotion effectiveness...")
        try:
            # Calculate sales lift during promotions for each store
            promo_lift = self.data.groupby('Store').apply(lambda x: 
                (x[x['Promo'] == 1]['Sales'].mean() / x[x['Promo'] == 0]['Sales'].mean()) - 1
            ).sort_values(ascending=False)

            # Identify top 10 stores with highest promo lift
            top_stores = promo_lift.head(10)

            # Visualize results
            plt.figure(figsize=(12, 6))
            sns.barplot(x=top_stores.index, y=top_stores.values)
            plt.title('Top 10 Stores with Highest Promotion Lift')
            plt.xlabel('Store ID')
            plt.ylabel('Sales Lift (%)')
            plt.savefig('../notebooks/figures/promo_effectiveness.png')
            plt.show()

            logging.info(f"Top 5 stores for promo effectiveness: {top_stores.head().to_dict()}")
            return top_stores
        except Exception as e:
            logging.error(f"Error in analyze_promo_effectiveness: {str(e)}")
            raise

    def analyze_store_opening_times(self):
        """
        Explore trends in customer behaviour during store opening and closing times.
        """
        logging.info("Analyzing store opening times...")
        try:
            # Add hour column
            self.data['Hour'] = pd.to_datetime(self.data['Date']).dt.hour

            # Calculate average sales and customers by hour
            hourly_data = self.data.groupby('Hour')[['Sales', 'Customers']].mean()

            # Visualize results
            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax2 = ax1.twinx()

            sns.lineplot(data=hourly_data, x=hourly_data.index, y='Sales', ax=ax1, color='b', label='Sales')
            sns.lineplot(data=hourly_data, x=hourly_data.index, y='Customers', ax=ax2, color='r', label='Customers')

            ax1.set_xlabel('Hour of Day')
            ax1.set_ylabel('Average Sales')
            ax2.set_ylabel('Average Customers')
            plt.title('Sales and Customer Trends by Hour')
            fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

            plt.savefig('../notebooks/figures/store_opening_times.png')
            plt.show()

            logging.info("Store opening times analysis completed.")
            return hourly_data
        except Exception as e:
            logging.error(f"Error in analyze_store_opening_times: {str(e)}")
            raise

    def analyze_weekday_weekend_sales(self):
        """
        Determine if stores open on all weekdays experience different sales patterns on weekends.
        """
        logging.info("Analyzing weekday vs weekend sales...")
        try:
            # Create weekend flag
            self.data['IsWeekend'] = self.data['DayOfWeek'].isin([6, 7]).astype(int)

            # Identify stores open all days
            stores_open_all_days = self.data.groupby('Store')['Open'].nunique() == 1

            # Calculate average sales for weekdays and weekends for stores open all days
            sales_patterns = self.data[self.data['Store'].isin(stores_open_all_days[stores_open_all_days].index)].groupby(['Store', 'IsWeekend'])['Sales'].mean().unstack()
            sales_patterns['WeekendRatio'] = sales_patterns[1] / sales_patterns[0]

            # Visualize results
            plt.figure(figsize=(12, 6))
            sns.histplot(sales_patterns['WeekendRatio'], bins=30, kde=True)
            plt.title('Distribution of Weekend to Weekday Sales Ratio')
            plt.xlabel('Weekend to Weekday Sales Ratio')
            plt.savefig('../notebooks/figures/weekday_weekend_sales.png')
            plt.show()

            logging.info(f"Average weekend to weekday sales ratio: {sales_patterns['WeekendRatio'].mean():.2f}")
            return sales_patterns
        except Exception as e:
            logging.error(f"Error in analyze_weekday_weekend_sales: {str(e)}")
            raise

    def analyze_assortment_impact(self):
        """
        Analyse how different assortment types affect sales.
        """
        logging.info("Analyzing assortment impact on sales...")
        try:
            # Calculate average sales by assortment type
            assortment_sales = self.data.groupby('Assortment')['Sales'].mean().sort_values(ascending=False)

            # Visualize results
            plt.figure(figsize=(10, 6))
            sns.barplot(x=assortment_sales.index, y=assortment_sales.values)
            plt.title('Average Sales by Assortment Type')
            plt.xlabel('Assortment Type')
            plt.ylabel('Average Sales')
            plt.savefig('../notebooks/figures/assortment_impact.png')
            plt.show()

            logging.info(f"Assortment sales ranking: {assortment_sales.to_dict()}")
            return assortment_sales
        except Exception as e:
            logging.error(f"Error in analyze_assortment_impact: {str(e)}")
            raise

    def analyze_competitor_proximity(self):
        """
        Evaluate how the distance to the nearest competitor impacts sales.
        Assess if this impact changes for stores in city centres.
        """
        logging.info("Analyzing competitor proximity impact...")
        try:
            # Create bins for competition distance
            self.data['CompetitionDistanceBin'] = pd.cut(self.data['CompetitionDistance'], 
                                                         bins=[0, 1000, 5000, 10000, np.inf], 
                                                         labels=['<1km', '1-5km', '5-10km', '>10km'])

            # Calculate average sales by competition distance bin
            sales_by_distance = self.data.groupby('CompetitionDistanceBin')['Sales'].mean()

            # Identify city centre stores (assuming those with competition within 1km are in city centres)
            city_centre_stores = self.data[self.data['CompetitionDistance'] <= 1000]['Store'].unique()

            # Calculate sales impact for city centre stores
            city_centre_impact = self.data[self.data['Store'].isin(city_centre_stores)].groupby('CompetitionDistanceBin')['Sales'].mean()

            # Visualize results
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            sns.barplot(x=sales_by_distance.index, y=sales_by_distance.values, ax=ax1)
            ax1.set_title('Average Sales by Competitor Distance')
            ax1.set_xlabel('Distance to Nearest Competitor')
            ax1.set_ylabel('Average Sales')

            sns.barplot(x=city_centre_impact.index, y=city_centre_impact.values, ax=ax2)
            ax2.set_title('Average Sales by Competitor Distance (City Centre Stores)')
            ax2.set_xlabel('Distance to Nearest Competitor')
            ax2.set_ylabel('Average Sales')

            plt.tight_layout()
            plt.savefig('../notebooks/figures/competitor_proximity_impact.png')
            plt.show()

            logging.info(f"Sales by competitor distance: {sales_by_distance.to_dict()}")
            logging.info(f"City centre stores sales by competitor distance: {city_centre_impact.to_dict()}")
            return sales_by_distance, city_centre_impact
        except Exception as e:
            logging.error(f"Error in analyze_competitor_proximity: {str(e)}")
            raise

    def analyze_new_competitor_impact(self):
        """
        Analyse how the opening or reopening of competitors affects sales for nearby Rossmann stores.
        Focus on stores where competitor distance information was initially unavailable but became available later.
        """
        logging.info("Analyzing new competitor impact...")
        try:
            # Identify stores with new competitors
            stores_with_new_competitors = self.data[
                (self.data['CompetitionOpenSinceYear'].notnull()) & 
                (self.data['CompetitionOpenSinceYear'] >= self.data['Date'].dt.year.min())
            ]['Store'].unique()

            # Calculate sales before and after new competitor for each store
            impact_data = []
            for store in stores_with_new_competitors:
                store_data = self.data[self.data['Store'] == store]
                competitor_data = store_data[store_data['CompetitionOpenSinceYear'].notnull()].iloc[0]
                
                # Extract year and month, then create a datetime object
                year = int(competitor_data['CompetitionOpenSinceYear'])
                month = int(competitor_data['CompetitionOpenSinceMonth'])
                competitor_open_date = pd.Timestamp(year=year, month=month, day=1)
                
                logging.debug(f"Store {store} - Competitor open date: {competitor_open_date}")
                
                before_sales = store_data[store_data['Date'] < competitor_open_date]['Sales'].mean()
                after_sales = store_data[store_data['Date'] >= competitor_open_date]['Sales'].mean()
                
                impact_data.append({
                    'Store': store,
                    'BeforeSales': before_sales,
                    'AfterSales': after_sales,
                    'SalesChange': (after_sales - before_sales) / before_sales if before_sales != 0 else 0
                })

            impact_df = pd.DataFrame(impact_data).sort_values('SalesChange')

            # Visualize results
            plt.figure(figsize=(12, 6))
            sns.scatterplot(data=impact_df, x='BeforeSales', y='AfterSales')
            plt.plot([impact_df['BeforeSales'].min(), impact_df['BeforeSales'].max()], 
                    [impact_df['BeforeSales'].min(), impact_df['BeforeSales'].max()], 
                    'r--', label='No Change Line')
            plt.title('Sales Before vs After New Competitor')
            plt.xlabel('Average Sales Before New Competitor')
            plt.ylabel('Average Sales After New Competitor')
            plt.legend()
            plt.savefig('../notebooks/figures/new_competitor_impact.png')
            plt.show()

            logging.info(f"Average sales change after new competitor: {impact_df['SalesChange'].mean():.2%}")
            return impact_df
        except Exception as e:
            logging.error(f"Error in analyze_new_competitor_impact: {str(e)}")
            raise

    def run_all_analyses(self):
        """
        Run all sales analysis functions and compile results.
        """
        logging.info("Running all sales analyses...")
        results = {}
        try:
            results['promo_effectiveness'] = self.analyze_promo_effectiveness()
            results['store_opening_times'] = self.analyze_store_opening_times()
            results['weekday_weekend_sales'] = self.analyze_weekday_weekend_sales()
            results['assortment_impact'] = self.analyze_assortment_impact()
            results['competitor_proximity'] = self.analyze_competitor_proximity()
            results['new_competitor_impact'] = self.analyze_new_competitor_impact()
            logging.info("All sales analyses completed successfully.")
            return results
        except Exception as e:
            logging.error(f"Error in run_all_analyses: {str(e)}")
            raise

if __name__ == "__main__":
     # Example usage
    from load_data import DataLoader
    from clean_data import DataCleaner

    loader = DataLoader()
    loader.load_data('../data/raw/train.csv', '../data/raw/test.csv', '../data/raw/store.csv')
    merged_train, merged_test = loader.merge_data()

    cleaner = DataCleaner()
    cleaned_train = cleaner.preprocess_data(merged_train)
    cleaned_test = cleaner.preprocess_data(merged_test)
    try:
      
        analyzer = SalesAnalyzer(merged_train)
        results = analyzer.run_all_analyses()
        print("Analysis complete. Results and visualizations have been saved.")
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")