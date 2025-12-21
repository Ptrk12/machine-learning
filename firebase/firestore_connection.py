import logging
import os
from google.cloud import firestore
from google.oauth2 import service_account
import json

class FirebaseConnection:
    def __init__(self):
        self.env_val = os.environ.get("FIREBASE_CREDENTIALS_JSON")
        self.cient = None
        self.creds_dict = None
        
        if not self.env_val:
            logging.error("Missing variable: FIREBASE_CREDENTIALS_JSON")
            raise ValueError("Environment variable FIREBASE_CREDENTIALS_JSON is not set")
        
        try:
            if os.path.exists(self.env_val):
                with open(self.env_val,'r') as f:
                    self.creds_dict = json.load(f)
            else:
                self.creds_dict = json.loads(self.env_val)
        except Exception as e:
            logging.error(f"Error loading Firebase credentials: {e}")
            raise
        
    def __enter__(self):
        try:
            credentials = service_account.Credentials.from_service_account_info(self.creds_dict)
            self.client = firestore.Client(credentials=credentials,project=credentials.project_id)
            return self.client
        except Exception as e:
            logging.error(f"Failed to create firestore client: {e}")
            raise
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.client:
            self.client.close()