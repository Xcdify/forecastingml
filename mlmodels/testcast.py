import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
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

# Step 4: Create sequences for LSTM
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)

# Step 5: Build and train LSTM model
def build_lstm_model(train_data, seq_length):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    X_train = create_sequences(train_data, seq_length)
    y_train = train_data[seq_length:]
    
    model.fit(X_train, y_train, epochs=100, verbose=0)
    return model

# Step 6: Make predictions
def make_predictions(model, data, seq_length):
    X = create_sequences(data, seq_length)
    return model.predict(X).flatten()

# Step 7: Evaluate model performance
def evaluate_model(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(max(0, mse))  # Ensure non-negative value before sqrt
    mae = mean_absolute_error(actual, predicted)
    return rmse, mae

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
            
            if len(ts_data) > 10:  # Ensure we have at least some data to work with
                # Normalize the data
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(ts_data.values.reshape(-1, 1)).flatten()
                
                # Define sequence length
                seq_length = min(7, len(scaled_data) - 1)  # Use weekly seasonality if possible
                
                # Split data into train and test sets
                train_size = int(len(scaled_data) * 0.8)
                train_data = scaled_data[:train_size]
                test_data = scaled_data[train_size:]
                
                # Build and train LSTM model
                model = build_lstm_model(train_data, seq_length)
                
                # Make predictions
                train_predict = make_predictions(model, train_data, seq_length)
                test_predict = make_predictions(model, test_data, seq_length)
                
                # Inverse transform predictions
                train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1)).flatten()
                test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1)).flatten()
                
                # Evaluate model performance
                rmse, mae = evaluate_model(ts_data.values[train_size+seq_length:], test_predict)
                
                results[item] = {
                    'rmse': rmse,
                    'mae': mae,
                    'actual': ts_data,
                    'train_predict': train_predict,
                    'test_predict': test_predict,
                    'seq_length': seq_length
                }
            else:
                print(f"Insufficient data for item {item}. Skipping.")
        except Exception as e:
            print(f"Error processing item {item}: {str(e)}")
    
    # Print evaluation metrics
    print("\nEvaluation Metrics:")
    for item, result in results.items():
        print(f"Item: {item}")
        print(f"  RMSE: {result['rmse']:.2f}")
        print(f"  MAE: {result['mae']:.2f}")
    
    # Plot actual vs predicted for a sample item
    if results:
        sample_item = list(results.keys())[0]
        result = results[sample_item]
        
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Forecast vs Actual', 'Residuals'))
        
        # Forecast vs Actual
        fig.add_trace(go.Scatter(x=result['actual'].index, y=result['actual'].values,
                                 mode='lines', name='Actual', line=dict(color='blue')), row=1, col=1)
        
        train_index = result['actual'].index[:len(result['train_predict'])]
        fig.add_trace(go.Scatter(x=train_index[result['seq_length']:], y=result['train_predict'],
                                 mode='lines', name='Train Forecast', line=dict(color='green')), row=1, col=1)
        
        test_index = result['actual'].index[-len(result['test_predict']):]
        fig.add_trace(go.Scatter(x=test_index, y=result['test_predict'],
                                 mode='lines', name='Test Forecast', line=dict(color='red')), row=1, col=1)
        
        # Residuals
        residuals = result['actual'].values[-len(result['test_predict']):] - result['test_predict']
        fig.add_trace(go.Scatter(x=test_index, y=residuals,
                                 mode='lines', name='Residuals', line=dict(color='purple')), row=2, col=1)
        
        fig.update_layout(height=800, title_text=f"Demand Forecast Analysis for Item {sample_item}")
        fig.show()
    else:
        print("No items with sufficient data for forecasting.")

# Run the demand forecasting system
run_demand_forecasting('itemusagedetailssmall.csv')