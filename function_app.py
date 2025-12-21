import os
import sys
import types
if '__main__' not in sys.modules:
    sys.modules['__main__'] = types.ModuleType('__main__')
import azure.functions as func
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import requests
import logging
from datetime import  timedelta
from sql.sql_repository import get_device_location, get_serial_number_by_device_id
from firebase.firestore_repository import get_device_measurements

app = func.FunctionApp()

lstm_model = None
bilstm_model = None
attn_lstm_model = None
rf_model = None
scaler = None

COLUMNS = ['PM25', 'PM10', 'temp_c', 'humidity_percent', 'pressure_hpa']

COLUMN_MAPPING = {
    'temperature': 'temp_c',
    'humidity': 'humidity_percent',
    'pressure': 'pressure_hpa',
    'pm2_5': 'PM25',
    'pm10': 'PM10'
}

DEFAULT_LAT = 50.0647
DEFAULT_LON = 19.9450

def load_artifacts():
    global lstm_model, bilstm_model, scaler, attn_lstm_model, rf_model
    base_path = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_path, "models")

    if scaler is None:
        scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
        logging.info("Scaler loaded successfully.")
    
    if lstm_model is None:
        lstm_model = tf.keras.models.load_model(os.path.join(models_dir, "lstm_model.keras"))
        logging.info("LSTM model loaded successfully.")
        
    if bilstm_model is None:
        bilstm_model = tf.keras.models.load_model(os.path.join(models_dir, "bilstm_model.keras"))
        logging.info("Bi-LSTM model loaded successfully.")
        
    if attn_lstm_model is None:
        attn_lstm_model = tf.keras.models.load_model(os.path.join(models_dir, "attn_lstm_model.keras"))
        logging.info("Attention LSTM model loaded successfully.")

    if rf_model is None:
        rf_model = joblib.load(os.path.join(models_dir, "rf_model.pkl"))
        logging.info("Random Forest model loaded successfully.")


def fetch_hybrid_data(device_id):
    device_serial_number = get_serial_number_by_device_id(device_id)
    
    if device_serial_number is None:
        raise ValueError(f"Device ID {device_id} not found in SQL database.")
    
    logging.info(f"Checking for device data in Firestore for ID: {device_id}")
    df_device = get_device_measurements(device_serial_number)
    
    has_device_data = df_device is not None and not df_device.empty
    
    if has_device_data:
        logging.info("Device data detected")
        sql_coords = get_device_location(device_id)
        lat, lon = sql_coords if sql_coords else (DEFAULT_LAT, DEFAULT_LON)
    else:
        lat, lon = DEFAULT_LAT, DEFAULT_LON
        logging.info("No device data found. Using default coordinates.")

    url_air = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=pm10,pm2_5&past_days=2"
    url_weather = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,surface_pressure&past_days=2"
    
    try:
        r_air = requests.get(url_air).json()
        r_weather = requests.get(url_weather).json()

        if 'error' in r_air or 'error' in r_weather:
            raise Exception("OpenMeteo API Error")
        
        s_pm25 = pd.Series(r_air['hourly']['pm2_5'], index=pd.to_datetime(r_air['hourly']['time']), name='PM25')
        s_pm10 = pd.Series(r_air['hourly']['pm10'], index=pd.to_datetime(r_air['hourly']['time']), name='PM10')
        s_temp = pd.Series(r_weather['hourly']['temperature_2m'], index=pd.to_datetime(r_weather['hourly']['time']), name='temp_c')
        s_hum = pd.Series(r_weather['hourly']['relative_humidity_2m'], index=pd.to_datetime(r_weather['hourly']['time']), name='humidity_percent')
        s_pres = pd.Series(r_weather['hourly']['surface_pressure'], index=pd.to_datetime(r_weather['hourly']['time']), name='pressure_hpa')

        df_api = pd.concat([s_pm25, s_pm10, s_temp, s_hum, s_pres], axis=1)
        df_api.index = df_api.index.tz_localize(None) # Usuwamy strefę czasową dla zgodności
        df_api.sort_index(inplace=True)

        current_time = pd.Timestamp.utcnow().tz_localize(None)
        
        if has_device_data:
            df_combined = df_device.combine_first(df_api)
        else:
            df_combined = df_api

        df_final = df_combined[df_combined.index <= current_time].tail(24)
        df_final = df_final.ffill().bfill()

        if len(df_final) < 24:
            raise ValueError(f"Insufficient data. Found {len(df_final)} rows after merge.")
        
        return df_final[COLUMNS].values, df_final.index[-1]
        
    except Exception as e:
        logging.error(f"Process Error: {e}")
        raise e

@app.route(route="predict_pollution", auth_level=func.AuthLevel.ANONYMOUS)
def predict_pollution(req: func.HttpRequest) -> func.HttpResponse:
    try:
        load_artifacts()
        selected_model = None
        
        model_type = req.params.get('model', 'lstm').lower()
        device_id = req.params.get('deviceId')
        hours_to_predict = int(req.params.get('hours', 24))
                        
        if model_type == 'bilstm':
            selected_model = bilstm_model
        elif model_type == 'lstm':
            selected_model = lstm_model
        elif model_type == 'attn_lstm':
            selected_model = attn_lstm_model
        elif model_type == 'rf':
            selected_model = rf_model
        
        raw_data, last_time = fetch_hybrid_data(device_id)
        
        current_input_scaled = scaler.transform(raw_data)
        current_batch = np.expand_dims(current_input_scaled, axis=0)

        predictions_scaled = []
        future_timestamps = []

        for _ in range(hours_to_predict):
            next_pred = selected_model.predict(current_batch, verbose=0)
            predictions_scaled.append(next_pred[0])
            last_time += timedelta(hours=1)
            future_timestamps.append(last_time.isoformat())
            next_pred_reshaped = np.reshape(next_pred, (1, 1, 5))
            current_batch = np.append(current_batch[:, 1:, :], next_pred_reshaped, axis=1)

        predictions_real = scaler.inverse_transform(predictions_scaled)
        
        results = []
        for i in range(len(predictions_real)):
            row = predictions_real[i]
            results.append({
                "time": future_timestamps[i],
                "PM25": round(float(row[0]), 2),
                "PM10": round(float(row[1]), 2),
                "temp_c": round(float(row[2]), 1),
                "humidity": round(float(row[3]), 1),
                "pressure": round(float(row[4]), 1)
            })

        return func.HttpResponse(json.dumps({
            "model": "Bi-LSTM" if 'bi' in model_type else "LSTM",
            "hours": hours_to_predict,
            "predictions": results
        }), mimetype="application/json")

    except Exception as e:
        return func.HttpResponse(str(e), status_code=500)