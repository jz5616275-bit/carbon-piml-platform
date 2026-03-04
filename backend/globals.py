import os
from pymongo import MongoClient


MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "carbon_piml")

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=1500)
db = client[MONGO_DB]