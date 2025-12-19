import azure.functions as func
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import requests
import os
from datetime import datetime, timedelta

app = func.FunctionApp()

lstm_model = None
bilstm_model = None
scaler = None
COLUMNS = ['PM25', 'PM10', 'temp_c', 'humidity_percent', 'pressure_hpa']

def load_artifacts():
    global lstm_model, bilstm_model, scaler
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_path, "models")

    if scaler is None:
        scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    
    if lstm_model is None:
        lstm_model = tf.keras.models.load_model(os.path.join(models_dir, "lstm_model.keras"))
        
    if bilstm_model is None:
        bilstm_model = tf.keras.models.load_model(os.path.join(models_dir, "bilstm_model.keras"))

def fetch_weather_data():
    lat, lon = 50.0647, 19.9450 
    
    url_air = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=pm10,pm2_5&past_days=2"
    url_weather = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,surface_pressure&past_days=2"
    
    r_air = requests.get(url_air).json()
    r_weather = requests.get(url_weather).json()

    if 'error' in r_air or 'error' in r_weather:
        raise Exception("API Error")
    
    s_pm25 = pd.Series(r_air['hourly']['pm2_5'], index=pd.to_datetime(r_air['hourly']['time']), name='PM25')
    s_pm10 = pd.Series(r_air['hourly']['pm10'], index=pd.to_datetime(r_air['hourly']['time']), name='PM10')
    
    s_temp = pd.Series(r_weather['hourly']['temperature_2m'], index=pd.to_datetime(r_weather['hourly']['time']), name='temp_c')
    s_hum = pd.Series(r_weather['hourly']['relative_humidity_2m'], index=pd.to_datetime(r_weather['hourly']['time']), name='humidity_percent')
    s_pres = pd.Series(r_weather['hourly']['surface_pressure'], index=pd.to_datetime(r_weather['hourly']['time']), name='pressure_hpa')

    df = pd.concat([s_pm25, s_pm10, s_temp, s_hum, s_pres], axis=1)
    
    df.dropna(inplace=True)
    df.sort_index(inplace=True)
    
    df_past = df[df.index <= pd.Timestamp.now()].tail(24)
    
    if len(df_past) < 24:
        raise ValueError(f"Insufficient data points. Found only {len(df_past)}")
    
    return df_past.values, df_past.index[-1]

@app.route(route="predict_pollution", auth_level=func.AuthLevel.ANONYMOUS)
def predict_pollution(req: func.HttpRequest) -> func.HttpResponse:
    try:
        load_artifacts()

        model_type = req.params.get('model', 'lstm').lower()
        hours_to_predict = int(req.params.get('hours', 24))

        selected_model = bilstm_model if 'bi' in model_type else lstm_model
        
        raw_data, last_time = fetch_weather_data()
        
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

        response_payload = {
            "model": "Bi-LSTM" if 'bi' in model_type else "LSTM",
            "hours": hours_to_predict,
            "predictions": results
        }

        return func.HttpResponse(
            json.dumps(response_payload),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        return func.HttpResponse(str(e), status_code=500)