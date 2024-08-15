from app.config.environments import ENV_MOCK_DATA_FOLDER
import torch

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_device_info():
    print(torch.__version__)
    print(torch.version.cuda)
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Current device: {torch.cuda.current_device()}" if torch.cuda.is_available() else "No CUDA device available")



# Constantes
TOKEN_HUGGINGFACE = ENV_MOCK_DATA_FOLDER# Reemplazar con tu token
MODEL_NAME = "facebook/chameleon-7b"
LEARNING_RATE = 5e-5
BATCH_SIZE = 2
MAX_STEPS_PER_EPOCH = 5  # Ajusta este número según sea necesario
EPOCHS = 3