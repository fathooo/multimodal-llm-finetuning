import os
import pandas as pd
from app.data.dataset import TextDataset
from colorama import Fore
from colorama import Fore, Style
from app.utils.colorama_utils import print_message

def load_dataset(file_path, file_format):
    if not os.path.exists(file_path):
        print_message(f"File {file_path} does not exist.", Fore.RED)
        return None
    if file_format == 'csv':
        df = pd.read_csv(file_path)
    elif file_format == 'parquet':
        df = pd.read_parquet(file_path)
    else:
        print_message(f"Unsupported file format: {file_format}", Fore.RED)
        return None
    required_columns = ['instruction', 'input', 'output']
    if not all(column in df.columns for column in required_columns):
        print_message(f"File {file_path} does not contain the required columns: {required_columns}", Fore.RED)
        return None
    if len(df) >= 3:
        preview = df.head(3)
        print_message("Preview of the first 3 rows of the dataset:", Fore.GREEN)
        print(Fore.GREEN + preview.to_string(index=False))
        print(Style.RESET_ALL)
    return df


def split_dataset(df, tokenizer, device, split_ratio=0.8):
    texts = df[['instruction', 'input', 'output']].apply(lambda x: x['instruction'] + x['input'], axis=1).tolist()
    split_idx = int(len(texts) * split_ratio)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    train_dataset = TextDataset(pd.DataFrame(train_texts, columns=['text']), tokenizer, device)
    val_dataset = TextDataset(pd.DataFrame(val_texts, columns=['text']), tokenizer, device)
    return train_dataset, val_dataset
