import pandas as pd
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from app.utils.huggingface import hf_login

def prepare_and_tokenize_dataset(token, processor):
    # Iniciar sesión en Hugging Face
    hf_login(token)

    # Cargar el dataset de Alpaca en español desde Hugging Face
    dataset = load_dataset('bertin-project/alpaca-spanish')

    # Dividir el dataset en 90% entrenamiento y 10% evaluación
    train_test_split = dataset['train'].train_test_split(test_size=0.1)
    tokenized_datasets = DatasetDict({
        'train': train_test_split['train'],
        'test': train_test_split['test']
    })

    # Función de tokenización utilizando el procesador de Chameleon
    def tokenize_function(examples):
        concatenated_texts = [
            instruction + " " + input_text + " " + output_text
            for instruction, input_text, output_text in zip(examples["instruction"], examples["input"], examples["output"])
        ]
        # Tokenizar el texto y usar los input_ids como etiquetas
        tokenized_inputs = processor(concatenated_texts, padding="max_length", truncation=True, max_length=512)
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
        return tokenized_inputs

    # Tokenizar el dataset dividido
    tokenized_datasets = tokenized_datasets.map(tokenize_function, batched=True)

    # Cargar los datos con DataLoader y collate_fn para manejar el batching
    selected_indices = list(range(1))  # Seleccionar los índices del 0 al 9
    train_loader = DataLoader(tokenized_datasets["train"].select(selected_indices), batch_size=1, shuffle=True)

    return tokenized_datasets, train_loader