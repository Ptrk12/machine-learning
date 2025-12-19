import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

CONFIG = {
    'csv_path': '../data/merged_data.csv',
    'time_steps': 24,
    'test_split': 0.2,
    'epochs': 50,
    'batch_size': 32,
    'patience': 5
}

def load_and_process_data(csv_path, time_steps):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    # POPRAWKA 1: low_memory=False usuwa ostrzeżenie o mieszanych typach
    df = pd.read_csv(csv_path, low_memory=False)
    
    cols = ['PM25', 'PM10', 'temp_c', 'humidity_percent', 'pressure_hpa']
    
    # Konwersja na liczby (zamienia błędy na NaN)
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    initial_len = len(df)
    df.dropna(subset=cols, inplace=True)
    
    if len(df) == 0:
        raise ValueError("Error: All data was dropped! Check your CSV file format.")

    if 'timestamp' in df.columns:
        # POPRAWKA 2: format='mixed' radzi sobie z ".000" i różnymi formatami dat
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        df.sort_values('timestamp', inplace=True)
        
    data_values = df[cols].values.astype('float32')
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_values)
    
    X, y = [], []
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i : i + time_steps])
        y.append(scaled_data[i + time_steps])  

    return np.array(X), np.array(y), scaler, cols

def split_data(X, y, split_ratio):
    train_size = int(len(X) * (1 - split_ratio))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, y_train, X_test, y_test

def build_lstm_model(input_shape, n_outputs):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(n_outputs) 
    ], name="Standard_LSTM")
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_bilstm_model(input_shape, n_outputs):
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.2),
        Dense(n_outputs)
    ], name="Bidirectional_LSTM")
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_model(model, X_train, y_train, X_test, y_test, config):
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=config['patience'], 
        restore_best_weights=True
    )
    
    print(f"Starting training for {model.name}")
    history = model.fit(
        X_train, y_train,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )
    return history

if __name__ == "__main__":
    try:
        X, y, scaler, cols = load_and_process_data(CONFIG['csv_path'], CONFIG['time_steps'])
        X_train, y_train, X_test, y_test = split_data(X, y, CONFIG['test_split'])
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        n_outputs = y_train.shape[1]

        lstm_model = build_lstm_model(input_shape, n_outputs)
        bilstm_model = build_bilstm_model(input_shape, n_outputs)

        history_lstm = train_model(lstm_model, X_train, y_train, X_test, y_test, CONFIG)
        history_bilstm = train_model(bilstm_model, X_train, y_train, X_test, y_test, CONFIG)

        plt.figure(figsize=(10, 5))
        plt.plot(history_lstm.history['val_loss'], label='LSTM Val Loss', linestyle='--')
        plt.plot(history_bilstm.history['val_loss'], label='Bi-LSTM Val Loss', linestyle='-')
        plt.title('Validation Loss Comparison (MSE)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        lstm_filename = 'lstm_model.keras'
        lstm_model.save(lstm_filename)
        print(f"Standard LSTM saved as: {lstm_filename}")

        bilstm_filename = 'bilstm_model.keras'
        bilstm_model.save(bilstm_filename)
        print(f"Bidirectional LSTM saved as: {bilstm_filename}")

        scaler_filename = 'scaler.pkl'
        joblib.dump(scaler, scaler_filename)
        print(f"Scaler saved as: {scaler_filename}")

    except Exception as e:
        print(f"ERROR: {e}")