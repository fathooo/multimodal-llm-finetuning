from app.config.environments import ENV_TOKEN_HUGGINGFACE
import torch
from colorama import Fore, Style, init

init()

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_device_info():
    print(Fore.CYAN + "PyTorch Version: " + Fore.YELLOW + torch.__version__ + Style.RESET_ALL)
    print(Fore.CYAN + "CUDA Version: " + Fore.YELLOW + (torch.version.cuda if torch.version.cuda else "N/A") + Style.RESET_ALL)
    print(Fore.CYAN + "CUDA available: " + Fore.YELLOW + str(torch.cuda.is_available()) + Style.RESET_ALL)
    if torch.cuda.is_available():
        print(Fore.CYAN + "Current device: " + Fore.YELLOW + str(torch.cuda.current_device()) + Style.RESET_ALL)
    else:
        print(Fore.RED + "No CUDA device available")


# Constantes
TOKEN_HUGGINGFACE = ENV_TOKEN_HUGGINGFACE# Reemplazar con tu token
MODEL_NAME = "facebook/chameleon-7b"
LEARNING_RATE = 5e-5
BATCH_SIZE = 2
MAX_STEPS_PER_EPOCH = 5  # Ajusta este número según sea necesario
EPOCHS = 3