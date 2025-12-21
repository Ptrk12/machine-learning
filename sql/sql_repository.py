
import logging

from sql.sql_connection import SQLConnection

def get_device_location(device_id):
    try:
        with SQLConnection() as cursor:
            query = "SELECT Latitude,Longitude FROM Devices Where Id = ?"
            cursor.execute(query, (device_id,))
            row = cursor.fetchone()
            
            if row and row.Latitude is not None and row.Longitude is not None:
                return float(row.Latitude), float(row.Longitude)
            return None
    except Exception as e:
        logging.error(f"Error retrieving device location: {e}")
        return None
    
def get_serial_number_by_device_id(device_id):
    try:
        with SQLConnection() as cursor:
            query = "SELECT SerialNumber FROM Devices WHERE Id = ?"
            cursor.execute(query, (device_id,))
            row = cursor.fetchone()
            
            if row and row.SerialNumber is not None:
                return row.SerialNumber
            return None
    except Exception as e:
        logging.error(f"Error retrieving device serial number: {e}")
        return None