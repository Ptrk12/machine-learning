import azure.functions as func
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import requests
import os
import logging
import pyodbc
from datetime import datetime, timedelta, timezone
from google.cloud import firestore
from google.oauth2 import service_account

app = func.FunctionApp()

lstm_model = None
bilstm_model = None
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
    global lstm_model, bilstm_model, scaler
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

def get_firestore_client():
    env_val = os.environ.get("FIREBASE_CREDENTIALS_JSON")
    
    if not env_val:
        logging.error("Missing variable: FIREBASE_CREDENTIALS_JSON")
        return None
    
    full_config = None

    # STRATEGIA 1: Sprawdź czy to ścieżka do pliku
    if os.path.exists(env_val):
        try:
            with open(env_val, 'r') as f:
                full_config = json.load(f)
            logging.info(f"Loaded credentials from file: {env_val}")
        except Exception as e:
            logging.error(f"Error reading file {env_val}: {e}")
            return None
    
    else:
        try:
            full_config = json.loads(env_val)
        except Exception:
            logging.error("FIREBASE_CREDENTIALS_JSON is missing")
            return None

    try:
        if "Firebase" in full_config:
            creds_dict = full_config["Firebase"]
        else:
            creds_dict = full_config

        credentials = service_account.Credentials.from_service_account_info(creds_dict)
        return firestore.Client(credentials=credentials)
    except Exception as e:
        logging.error(f"Auth Error: {e}")
        return None

def get_device_location_sql(device_id):
    server = os.environ.get("SQL_SERVER", "localhost")
    database = os.environ.get("SQL_DATABASE", "Master")
    
    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={server};"
        f"DATABASE={database};"
        "Trusted_Connection=yes;"
    )

    try:
        with pyodbc.connect(conn_str, timeout=5) as conn:
            cursor = conn.cursor()
            query = "SELECT Latitude, Longitude FROM Devices WHERE DeviceId = ?"
            cursor.execute(query, device_id)
            row = cursor.fetchone()
            
            if row and row.Latitude is not None and row.Longitude is not None:
                logging.info(f"SQL: Found coordinates for device {device_id}: {row.Latitude}, {row.Longitude}")
                return float(row.Latitude), float(row.Longitude)
            
            logging.warning(f"SQL: Device {device_id} not found or coordinates missing.")
            return None
            
    except Exception as e:
        logging.error(f"SQL Connection Error: {e}")
        return None

def get_firestore_data(device_id, required_hours=24):
    try:
        db = get_firestore_client()
        if not db:
            return None

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=required_hours + 2)
        cutoff_timestamp = int(cutoff_time.timestamp())

        docs = db.collection('devices')\
                 .document(device_id)\
                 .collection('measurements')\
                 .where('timestamp', '>=', cutoff_timestamp)\
                 .stream()

        data_rows = []
        for doc in docs:
            d = doc.to_dict()
            ts_val = d.get('timestamp')
            if not ts_val: continue
            
            row = {'timestamp': pd.to_datetime(ts_val, unit='s')}
            params = d.get('parameters', [])
            if isinstance(params, list):
                for param_map in params:
                    for key, val in param_map.items():
                        if key in COLUMN_MAPPING:
                            row[COLUMN_MAPPING[key]] = val
            
            if len(row) > 1:
                data_rows.append(row)

        if not data_rows:
            return None

        df = pd.DataFrame(data_rows)
        for col in COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df['timestamp'] = df['timestamp'].dt.floor('h')
        df_grouped = df.groupby('timestamp')[COLUMNS].mean()
        
        return df_grouped

    except Exception as e:
        logging.error(f"Firestore Error: {e}")
        return None

def fetch_hybrid_data(device_id):

    logging.info(f"Checking for device data in Firestore for ID: {device_id}")
    df_firestore = get_firestore_data(device_id)
    
    has_device_data = df_firestore is not None and not df_firestore.empty

    if has_device_data:
        logging.info("Device data detected. Attempting to fetch coordinates from SQL...")
        sql_coords = get_device_location_sql(device_id)
        
        if sql_coords:
            lat, lon = sql_coords
            logging.info(f"Using SQL Device Coordinates: {lat}, {lon}")
        else:
            lat, lon = DEFAULT_LAT, DEFAULT_LON
            logging.warning(f"SQL lookup failed. Fallback to default coordinates: {lat}, {lon}")
    else:
        lat, lon = DEFAULT_LAT, DEFAULT_LON
        logging.info(f"No device data found. Using Default Krakow Coordinates: {lat}, {lon}")

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
        df_api.index = df_api.index.tz_localize(None)
        
        if has_device_data:
            if df_firestore.index.tz is not None:
                df_firestore.index = df_firestore.index.tz_localize(None)
            
            logging.info("Merging Firestore data into API data.")
            df_api.update(df_firestore)

        df_api.sort_index(inplace=True)
        current_time = pd.Timestamp.now()
        df_final = df_api[df_api.index <= current_time].tail(24)
        df_final = df_final.ffill().bfill()

        if len(df_final) < 24:
            raise ValueError(f"Insufficient data. Found {len(df_final)}")
        
        return df_final[COLUMNS].values, df_final.index[-1]
        
    except Exception as e:
        logging.error(f"Process Error: {e}")
        raise e

@app.route(route="predict_pollution", auth_level=func.AuthLevel.ANONYMOUS)
def predict_pollution(req: func.HttpRequest) -> func.HttpResponse:
    try:
        load_artifacts()
        model_type = req.params.get('model', 'lstm').lower()
        device_id = req.params.get('deviceId')
        hours_to_predict = int(req.params.get('hours', 24))
        
        selected_model = bilstm_model if 'bi' in model_type else lstm_model
        
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