import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def print_data_info(data, step):
    print(f"\n--- Data info after {step} ---")
    print(f"Shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    print(f"Non-null counts:\n{data.notnull().sum()}")
    print(f"Sample data:\n{data.head()}")

# Load the data
data = pd.read_csv('itemusage.csv')
print_data_info(data, "loading")

# Check if required columns exist
required_columns = ['ITEM_NO', 'WHSE_NAME', 'ACCTG_YEAR', 'ACCTG_PERIOD', 'QTY_SHIPPED']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

# Convert date columns to datetime
data['DATE'] = pd.to_datetime(data['ACCTG_YEAR'].astype(str) + '-' + data['ACCTG_PERIOD'].astype(str).str.zfill(2) + '-01')
print_data_info(data, "date conversion")

# Sort the data by date and item
data = data.sort_values(['ITEM_NO', 'DATE'])

# Create lag features
def create_lag_features(group, lag_list=[1, 2, 3]):
    for lag in lag_list:
        group[f'QTY_SHIPPED_LAG_{lag}'] = group['QTY_SHIPPED'].shift(lag)
    return group

# Create features for each item
data = data.groupby('ITEM_NO').apply(create_lag_features).reset_index(drop=True)
print_data_info(data, "creating lag features")

# Drop rows with NaN values (due to lag creation)
data_before_dropna = data.shape[0]
data = data.dropna()
data_after_dropna = data.shape[0]
print(f"Rows dropped due to NaN values: {data_before_dropna - data_after_dropna}")
print_data_info(data, "dropping NaN values")

# Check if we have enough data to proceed
if data.shape[0] < 10:  # Arbitrary threshold, adjust as needed
    raise ValueError(f"Insufficient data after preprocessing. Only {data.shape[0]} rows remaining.")

# Encode categorical variables
le_item = LabelEncoder()
le_whse = LabelEncoder()
data['ITEM_NO_ENCODED'] = le_item.fit_transform(data['ITEM_NO'])
data['WHSE_NAME_ENCODED'] = le_whse.fit_transform(data['WHSE_NAME'])

# Select features for the model
features = ['ITEM_NO_ENCODED', 'WHSE_NAME_ENCODED', 'ACCTG_YEAR', 'ACCTG_PERIOD', 
            'QTY_SHIPPED_LAG_1', 'QTY_SHIPPED_LAG_2', 'QTY_SHIPPED_LAG_3']
target = 'QTY_SHIPPED'

# Split the data into training and testing sets
X = data[features]
y = data[target]

try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
except ValueError as e:
    print(f"Error during train-test split: {str(e)}")
    print("Attempting to proceed with all data for training...")
    X_train, y_train = X, y
    X_test, y_test = None, None

# Initialize and train the XGBoost model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Make predictions and calculate metrics only if we have test data
if X_test is not None and y_test is not None:
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Root Mean Squared Error: {rmse}')
    print(f'Mean Absolute Error: {mae}')
else:
    print("No test set available. Skipping error metric calculation.")

# Function to forecast for the next period
def forecast_next_period(item_data, model, features):
    last_date = item_data['DATE'].max()
    next_date = last_date + pd.DateOffset(months=1)
    
    next_period = pd.DataFrame({
        'ITEM_NO_ENCODED': [item_data['ITEM_NO_ENCODED'].iloc[-1]],
        'WHSE_NAME_ENCODED': [item_data['WHSE_NAME_ENCODED'].iloc[-1]],
        'ACCTG_YEAR': [next_date.year],
        'ACCTG_PERIOD': [next_date.month],
        'QTY_SHIPPED_LAG_1': [item_data['QTY_SHIPPED'].iloc[-1]],
        'QTY_SHIPPED_LAG_2': [item_data['QTY_SHIPPED'].iloc[-2]],
        'QTY_SHIPPED_LAG_3': [item_data['QTY_SHIPPED'].iloc[-3]]
    })
    
    forecast = model.predict(next_period[features])
    return next_date, forecast[0]

# Generate forecasts for each item
forecasts = []
for item in data['ITEM_NO'].unique():
    item_data = data[data['ITEM_NO'] == item].sort_values('DATE')
    if len(item_data) >= 3:  # Ensure we have enough data for the item
        next_date, forecast = forecast_next_period(item_data, model, features)
        forecasts.append({
            'ITEM_NO': item,
            'DATE': next_date,
            'FORECAST': forecast
        })
    else:
        print(f"Insufficient data for item {item}. Skipping forecast.")

forecasts_df = pd.DataFrame(forecasts)
print("\nForecasts:")
print(forecasts_df)

# Plot actual vs predicted for a sample item, if we have enough data
if len(forecasts_df) > 0:
    sample_item = data['ITEM_NO'].value_counts().index[0]
    sample_data = data[data['ITEM_NO'] == sample_item].sort_values('DATE')

    plt.figure(figsize=(12, 6))
    plt.plot(sample_data['DATE'], sample_data['QTY_SHIPPED'], label='Actual')
    plt.plot(sample_data['DATE'], model.predict(sample_data[features]), label='Predicted')
    plt.scatter(forecasts_df[forecasts_df['ITEM_NO'] == sample_item]['DATE'], 
                forecasts_df[forecasts_df['ITEM_NO'] == sample_item]['FORECAST'], 
                color='red', label='Next Period Forecast')
    plt.title(f'Actual vs Predicted Quantities for Item {sample_item}')
    plt.xlabel('Date')
    plt.ylabel('Quantity Shipped')
    plt.legend()
    plt.show()
else:
    print("Not enough data to generate a plot.")