from dotenv import load_dotenv
import os

# Carga las variables de entorno desde el archivo .env
load_dotenv()

ENV_TOKEN_HUGGINGFACE = os.getenv('TOKEN_HUGGINGFACE', "https://huggingface.co/token/#here")
ENV_DOWNLOAD_DATA_DIR = os.getenv('DOWNLOAD_DATA_DIR', "app/datasets")
ENV_DOWNLOAD_DATA_EVAL_DIR = os.getenv('DOWNLOAD_DATA_EVAL_DIR', "app/datasets_eval")