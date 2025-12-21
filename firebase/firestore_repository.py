import datetime
import logging
import pandas as pd
from firebase.firestore_connection import FirebaseConnection

COLUMN_MAPPING = {
    'temperature': 'temp_c',
    'humidity': 'humidity_percent',
    'pressure': 'pressure_hpa',
    'pm2_5': 'PM25',
    'pm10': 'PM10'
}

COLUMNS = ['PM25', 'PM10', 'temp_c', 'humidity_percent', 'pressure_hpa']

def get_device_measurements(device_serial_number, required_hours=24):
    with FirebaseConnection() as db:
        try:
            cutoff_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=required_hours + 2)
            cutoff_timestamp = int(cutoff_time.timestamp())

            docs = db.collection('devices')\
                    .document(device_serial_number)\
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
        
            valid_cols = [c for c in COLUMNS if c in df.columns]
            for col in valid_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df['timestamp'] = df['timestamp'].dt.round('h')
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)

            df_grouped = df.groupby('timestamp')[valid_cols].mean()
            
            return df_grouped

        except Exception as e:
            logging.error(f"Firestore Error during data processing: {e}")
            return None