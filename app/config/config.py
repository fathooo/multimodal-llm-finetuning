from app.config.environments import ENV_TOKEN_HUGGINGFACE
import torch
from app.utils.colorama_utils import print_message
from colorama import Fore

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_device_info():
    print_message("PyTorch Version: " + torch.__version__, Fore.CYAN)
    print_message("CUDA Version: " + (torch.version.cuda if torch.version.cuda else "N/A"), Fore.CYAN)
    print_message("CUDA available: " + str(torch.cuda.is_available()), Fore.CYAN)
    if torch.cuda.is_available():
        print_message("Current device: " + str(torch.cuda.current_device()), Fore.CYAN)
    else:
        print_message("No CUDA device available", Fore.RED)

# Constants
TOKEN_HUGGINGFACE = ENV_TOKEN_HUGGINGFACE  # Replace with your token
MODEL_NAME = "facebook/chameleon-7b"
LEARNING_RATE = 5e-5
BATCH_SIZE = 2
MAX_STEPS_PER_EPOCH = 5  # Adjust this number as necessary
EPOCHS = 3