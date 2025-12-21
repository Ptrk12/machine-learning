import os
import pyodbc
import logging

class SQLConnection:
    def __init__(self):
        self.server = os.environ.get("SQL_SERVER")
        self.database = os.environ.get("SQL_DATABASE")
        self.conn = None
        self.cursor = None
    
    def __enter__(self):
        self.conn_str = (
            "DRIVER={ODBC Driver 17 for SQL Server};"
            f"SERVER={self.server};"
            f"DATABASE={self.database};"
            "Trusted_Connection=yes;"
        )
        
        try:
            self.conn = pyodbc.connect(self.conn_str, timeout=5)
            self.cursor = self.conn.cursor()
            return self.cursor
        except Exception as e:
            logging.error(f"Error connecting to SQL Server: {e}")
            raise
        
    def __exit__(self,exc_type,exc_val,exc_tb):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()