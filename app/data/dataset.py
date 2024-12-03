from torch.utils.data import Dataset
from datasets import load_dataset
import torch
import requests
import os
from colorama import Fore, Style, init
import pandas as pd
from PIL import Image

# Inicializar colorama
init(autoreset=True)

class CustomDataset(Dataset):
    def __init__(self, dataframe, processor, device):
        self.dataframe = dataframe
        self.processor = processor
        self.device = device

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        inputs = self.processor(
            text=row["text"] + " <image>",
            images=Image.open(requests.get(row['image_url'], stream=True).raw).convert("RGB").resize((512, 512)),
            return_tensors="pt",
            padding="max_length",
            max_length=512,
            truncation=True
        )
        inputs = {k: (v.to(self.device).to(torch.bfloat16) if k != 'input_ids' else v.to(self.device)) for k, v in inputs.items()}
        return inputs

class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, device):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        tokenized_text = self.tokenizer(
            row["instruction"] + row["input"],
            padding='max_length',
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )
        return {
            key: val.squeeze().to(self.device)
            for key, val in tokenized_text.items()
        }