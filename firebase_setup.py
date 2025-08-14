import os
import json
import base64
import firebase_admin
from firebase_admin import credentials, firestore

def get_firestore():
    encoded_key = os.environ.get("FIREBASE_SERVICE_KEY_B64")

    # Base64 decode et ve JSON'a çevir
    decoded_json = base64.b64decode(encoded_key).decode("utf-8")
    service_account_info = json.loads(decoded_json)

    if not firebase_admin._apps:
        cred = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(cred)

    return firestore.client()
