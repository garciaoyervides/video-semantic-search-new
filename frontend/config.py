from dotenv import load_dotenv
import os

load_dotenv()
API_ENDPOINT = os.getenv('API_ENDPOINT')
LAVIS_API_ENDPOINT = os.getenv('LAVIS_API_ENDPOINT')