# Rossmann Pharma Sales Forecasting
This project aims to explore customer purchasing behavior in various stores and predict store sales using machine learning and deep learning techniques. The analysis includes a thorough exploratory data analysis (EDA), model building, and serving predictions through an API.
### Table of Contents
1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Data Description](#data-description)
5. [Analysis Components](#analysis-components)
6. [Running Tests](#running-tests)
7. [Contributing](#contributing)
8. [License](#license)
## Project Overview
This project is designed to analyze and forecast sales for Rossmann Pharma stores. We aim to uncover customer purchasing behavior and build predictive models for daily sales. The analysis starts with data cleaning and exploration, followed by machine learning and deep learning models to make sales predictions. The results are then served through an API for real-time predictions.

### Key Objectives
Exploration of Customer Purchasing Behavior: Understand how various factors like promotions, holidays, store openings, and competitor presence affect purchasing behavior.
Sales Prediction: Build machine learning and deep learning models to predict future sales.
Model Serving: Provide an API interface for serving predictions in real-time.

## Project Structure
```
rossmann_pharma_sales_forecasting/
│
├── data/
│   └── raw/         # Processed data ready for analysis
├── src/
│   ├── __init__.py        # Makes src a module
│   ├── load_data.py       # Data Loading functions
│   ├── clean_data.py      # Data cleaning functions
│   ├── eda.py             # Functions for exploratory data analysis
|   └──utils.py            # Functions to plot visualizations
│
├── notebooks/             # Jupyter Notebooks for initial analysis
│   ├── __init__.py
│   ├── sales_analysis_eda.ipynb
│   └── model_building.ipynb
│
├── scripts/
│   └── __init__.py
│
├── tests/               # Tests for the modules
│   └── __init__.py
│
├── requirements.txt     # Python dependencies
├── README.md            # Project README
└── LICENSE              # License for the project
```

## Installation
To run this project, follow these steps to set up your environment:

1. **Clone the repository**: 
``` 
git clone https://github.com/your-username/rossmann_pharma_sales_forecasting.git 
```
2. **Navigate to the project directory**:
```
cd rossmann_pharma_sales_forecasting
```
3. **Create a virtual environment**:
```python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```
4. **Install the required packages**:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Place your data file in the `rossmann_pharma_sales_forecasting/data/` directory.

2. Run the main analysis script:
   ```
   python scripts/run_analysis.py
   ```

3. Open and run the Jupyter notebook for detailed exploratory data analysis:
   ```
   jupyter notebook notebooks/sales_analysis_eda.ipynb
   ```
## Data Description

The dataset includes information about stores, customers, state holidays, promotions, and competitors.
Key columns include:

- Promotional information - Promo, Promo2
- Competitors : CompetitionDistance , CompetitionOpenSince
- Customers and Sales: Customers, Sales, 
- Store details: Store, StoreType, Open, etc. 
- Holidays: StateHoliday, SchoolHoliday, etc.

## Analysis Components

1. **Data Loading and Preprocessing**: Handled by `src/load_data.py`
2. **Data Cleaning**: Handled by `src/clean_data.py`
3. **Basic Data Overview**: Handled by `src/utils.py`
4. **Exploratory Data Analysis**: Implemented in `src/eda.py`
5. **Sales Analysis**: Provided by `src/analyze_sales.py`

## Running Tests

To run the unit tests:

```
python -m unittest discover tests
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

For more information or support, please contact the project maintainers.