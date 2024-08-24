import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Step 1: Load and preprocess the data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['TRANS_DATE'] = pd.to_datetime(df['TRANS_DATE'])
    df = df.sort_values('TRANS_DATE')
    return df

# Step 2: Aggregate data by item and date
def aggregate_data(df):
    return df.groupby(['ITEM_NO', 'TRANS_DATE'])['QTY'].sum().reset_index()

# Step 3: Prepare time series data
def prepare_time_series(item_data):
    ts_data = item_data.set_index('TRANS_DATE')['QTY']
    ts_data = ts_data.resample('D').sum().fillna(0)
    return ts_data

# Step 4: Simple Moving Average model
def simple_moving_average(data, window):
    return data.rolling(window=window, min_periods=1).mean()

# Step 5: Make predictions
def make_predictions(data, window):
    ma = simple_moving_average(data, window)
    return ma.shift(1)  # Predict next day based on moving average

# Step 6: Evaluate model performance
def evaluate_model(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(max(0, mse))  # Ensure non-negative value before sqrt
    mae = mean_absolute_error(actual, predicted)
    return rmse, mae

# Step 7: Basic statistical analysis
def basic_stats(data):
    return {
        'count': len(data),
        'mean': data.mean(),
        'median': data.median(),
        'std': data.std(),
        'min': data.min(),
        'max': data.max()
    }

# Main function to run the demand forecasting system
def run_demand_forecasting(file_path):
    # Load and preprocess data
    df = load_and_preprocess_data(file_path)
    
    # Aggregate data
    agg_df = aggregate_data(df)
    
    # Get unique items
    items = agg_df['ITEM_NO'].unique()
    
    results = {}
    
    for item in items:
        try:
            item_data = agg_df[agg_df['ITEM_NO'] == item]
            ts_data = prepare_time_series(item_data)
            
            if len(ts_data) >= 7:  # If we have at least a week of data
                # Use simple moving average
                window = min(7, len(ts_data) // 2)  # Use up to 7 days, or half the data points
                predictions = make_predictions(ts_data, window)
                
                # Evaluate model performance
                rmse, mae = evaluate_model(ts_data[window:], predictions[window:])
                
                results[item] = {
                    'type': 'forecast',
                    'rmse': rmse,
                    'mae': mae,
                    'actual': ts_data,
                    'predictions': predictions,
                    'window': window
                }
            else:
                # For very limited data, just provide basic stats
                stats = basic_stats(ts_data)
                results[item] = {
                    'type': 'stats',
                    'stats': stats
                }
        except Exception as e:
            print(f"Error processing item {item}: {str(e)}")
    
    # Print results
    print("\nResults:")
    for item, result in results.items():
        print(f"\nItem: {item}")
        if result['type'] == 'forecast':
            print(f"  Forecast model used")
            print(f"  RMSE: {result['rmse']:.2f}")
            print(f"  MAE: {result['mae']:.2f}")
        else:
            print(f"  Basic statistics (insufficient data for forecasting)")
            for stat, value in result['stats'].items():
                print(f"    {stat}: {value:.2f}")
    
    # Plot for a sample forecasted item
    forecasted_items = [item for item, result in results.items() if result['type'] == 'forecast']
    if forecasted_items:
        sample_item = forecasted_items[45]
        result = results[sample_item]
        
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Forecast vs Actual', 'Residuals'))
        
        # Forecast vs Actual
        fig.add_trace(go.Scatter(x=result['actual'].index, y=result['actual'].values,
                                 mode='lines', name='Actual', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=result['predictions'].index, y=result['predictions'].values,
                                 mode='lines', name='Forecast', line=dict(color='red')), row=1, col=1)
        
        # Residuals
        residuals = result['actual'] - result['predictions']
        fig.add_trace(go.Scatter(x=residuals.index, y=residuals.values,
                                 mode='lines', name='Residuals', line=dict(color='purple')), row=2, col=1)
        
        fig.update_layout(height=800, title_text=f"Demand Forecast Analysis for Item {sample_item}")
        fig.show()
    else:
        print("No items with sufficient data for forecasting and plotting.")

# Run the demand forecasting system
run_demand_forecasting('itemusagedetails2018.csv')