import clip
import torch
import chromadb
import openai
import os
import sys
from dotenv import load_dotenv

global device
global model
global preprocess
global videos_collection
global segments_collection
global logs_collection
global vicuna_client
global yolo_model

load_dotenv()

ALLOWED_EXTENSIONS = ['mp4','mkv']
WHISPER_API_ENDPOINT  = os.getenv('WHISPER_API_ENDPOINT')
VICUNA_API_ENDPOINT  = os.getenv('VICUNA_API_ENDPOINT')
LAVIS_API_ENDPOINT = os.getenv('LAVIS_API_ENDPOINT')
DB_LOCATION =   "../db"

#Vicuna configuration
MAX_TOKENS = 900
LLM = "vicuna-13b-v1.5"
#LLM = "vicuna:13b"

#SETUP
try:
  # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)
    print("Device loaded: " + str(device))
except:
    print("Error loading model and data")
    exit()
try:
    client = chromadb.PersistentClient(path=DB_LOCATION)
    #client = chromadb.HttpClient(host='localhost', port=8000)
    videos_collection = client.get_or_create_collection(
        name ="videos",
        metadata={"hnsw:space": "cosine"}
        )
    segments_collection = client.get_or_create_collection(
        name ="segments_cos",
        metadata={"hnsw:space": "cosine"}
        )
    logs_collection = client.get_or_create_collection(
        name ="logs",
        metadata={"hnsw:space": "cosine"}
        )
    print("Database loaded") 
except:
    print("Database Error")
    exit()
try:
    vicuna_client = openai.OpenAI(
        api_key="EMPTY",
        base_url =VICUNA_API_ENDPOINT
        )

    print(VICUNA_API_ENDPOINT)    
except:
    print("Error connecting to Vicuna")
    exit()