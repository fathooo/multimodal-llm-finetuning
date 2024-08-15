from dotenv import load_dotenv
import os

# Carga las variables de entorno desde el archivo .env
load_dotenv()

ENV_TOKEN_HUGGINGFACE = os.getenv('TOKEN_HUGGINGFACE', "https://huggingface.co/token/#here")
ENV_MOCK_DATA_FOLDER = os.getenv('MOCK_DATA_FOLDER', "files/mock_data")