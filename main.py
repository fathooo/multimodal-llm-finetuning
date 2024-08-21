import os
import torch
from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
from app.config.environments import ENV_DOWNLOAD_DATA_DIR
from app.config.config import TOKEN_HUGGINGFACE, MODEL_NAME, get_device, print_device_info, LEARNING_RATE, MAX_STEPS_PER_EPOCH, EPOCHS
from app.data.download import download_datasets
from app.data.preprocess import load_dataset, split_dataset
from app.utils.huggingface import hf_login
from app.services.train import train_model
from app.utils.colorama_utils import initialize_colorama, print_message
from colorama import Fore  # Se agrega esta línea
from app.config.dataset_info import dataset_info_list

print("\n\n=============init=============\n\n")
initialize_colorama()
print_device_info()
device = get_device()
print_message(f"Using device: {device}", Fore.GREEN)

# Login to Hugging Face
hf_login(TOKEN_HUGGINGFACE)

# Download datasets if they don't exist
download_datasets(dataset_info_list, ENV_DOWNLOAD_DATA_DIR)

# Check if the file exists before loading
# Charge the first dataset
file_info = dataset_info_list[0]
file_path = os.path.join(ENV_DOWNLOAD_DATA_DIR, file_info['file_name'])
if not os.path.exists(file_path):
    print_message("File does not exist.", Fore.RED)

# Initialize processor to get tokenizer
processor = ChameleonProcessor.from_pretrained(MODEL_NAME)
tokenizer = processor.tokenizer

# Load local dataset
df = load_dataset(file_path, file_info['format'])
train_dataset, val_dataset = split_dataset(df, tokenizer, device)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")

if train_dataset and val_dataset:
    print("Dataset loaded successfully.")
    # Load model
    # model = ChameleonForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device)
    # # Train model
    # train_model(model, train_dataset, processor, device, LEARNING_RATE, MAX_STEPS_PER_EPOCH, EPOCHS)
    pass
else:
    print_message("Failed to load the dataset for training. Please check if the file exists and is accessible.", Fore.RED)