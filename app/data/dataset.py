from torch.utils.data import Dataset
import torch
from PIL import Image
import requests

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