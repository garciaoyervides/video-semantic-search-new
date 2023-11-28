import whisperx
import whisper
import torch
import openai
from dotenv import load_dotenv
import os

global device
global model
global client

API_ENDPOINT = os.getenv('API_ENDPOINT')
WHISPER_API_ENDPOINT = os.getenv('WHISPER_API_ENDPOINT')
BATCH_SIZE = 8 # max= 16 reduce if low on GPU mem
COMPUTE_TYPE = "int8" # best is "float16" change to "int8" if low on GPU mem (may reduce accuracy)

#Vicuna configuration
MAX_TOKENS = 900
LLM = "vicuna-13b-v1.5"
#SETUP 
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
#    model = whisper.load_model("medium", device)
#    #model = whisper.load_model("large-v2", device, compute_type=COMPUTE_TYPE)
#    #model = whisperx.load_model("large-v2", device, compute_type=COMPUTE_TYPE)
#    print(f"Whisper model loaded in {device}")
    print(f"Device loaded: {device}")

except:
    print("ERROR loading model and data")
    exit()

try:
    client = openai.OpenAI(
        api_key="EMPTY",
        base_url =WHISPER_API_ENDPOINT
        )

except:
    print("Error connecting to Vicuna")
    exit()