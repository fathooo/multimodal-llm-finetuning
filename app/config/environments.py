from dotenv import load_dotenv
import os

# Carga las variables de entorno desde el archivo .env
load_dotenv()

TOKEN_HUGGINGFACE = os.getenv('TOKEN_HUGGINGFACE', "https://huggingface.co/token/#here")
MOCK_DATA_FOLDER = os.getenv('MOCK_DATA_FOLDER', "files/mock_data")