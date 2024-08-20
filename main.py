#%%

import pandas as pd
from transformers import Trainer, TrainingArguments, ChameleonProcessor, ChameleonForConditionalGeneration
from app.config.config import TOKEN_HUGGINGFACE, MODEL_NAME, get_device, print_device_info, LEARNING_RATE, MAX_STEPS_PER_EPOCH, EPOCHS
from app.data.dataset import CustomDataset
from app.utils import hf_login
from app.services.train import train_model
from datasets import load_dataset, DatasetDict
import torch

print_device_info()
device = get_device()
print(f"Using device: {device}")

# Iniciar sesión en Hugging Face
hf_login("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")

# 1. Preparar el dataset
# Cargar el dataset de Alpaca en español desde Hugging Face
dataset = load_dataset('bertin-project/alpaca-spanish')
# Dividir el dataset en 90% entrenamiento y 10% evaluación
train_test_split = dataset['train'].train_test_split(test_size=0.1)
tokenized_datasets = DatasetDict({
    'train': train_test_split['train'],
    'test': train_test_split['test']
})
print(dataset)
#%%

#%%
# 2. Cargar el modelo Chameleon preentrenado y su procesador
processor = ChameleonProcessor.from_pretrained(MODEL_NAME)
model = ChameleonForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device)
#%%

#%%
# 3. Función de tokenización utilizando el procesador de Chameleon
def tokenize_function(examples):
    concatenated_texts = [
        instruction + " " + input_text + " " + output_text
        for instruction, input_text, output_text in zip(examples["instruction"], examples["input"], examples["output"])
    ]
    # Tokenizar el texto y usar los input_ids como etiquetas, ojo con el max_length que pueden quedarse datos fuera
    tokenized_inputs = processor(concatenated_texts, padding="max_length", truncation=True, max_length=256)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()  # Usar clone() en lugar de copy()
    return tokenized_inputs

# Tokenizar el dataset dividido (no el original)
# split_datasets = tokenized_datasets["train"].train_test_split(test_size=0.5)
tokenized_datasets = tokenized_datasets.map(tokenize_function, batched=True)
print(tokenized_datasets)
#%%

#%%
# 4. Configurar los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir="./results",              # Directorio donde se guardarán los resultados
    evaluation_strategy="epoch",        # Evaluación al final de cada época
    learning_rate=2e-5,                 # Tasa de aprendizaje
    per_device_train_batch_size=1,      # Tamaño del lote para entrenamiento
    per_device_eval_batch_size=1,       # Tamaño del lote para evaluación
    num_train_epochs=2,                 # Número de épocas de entrenamiento
    weight_decay=0.01,                  # Tasa de decaimiento del peso
    fp16=False,                          # Usar FP16
)

# 5. Inicializar el Trainer
trainer = Trainer(
    model=model,                        # Modelo que vamos a entrenar
    args=training_args,                 # Argumentos de entrenamiento
    train_dataset=tokenized_datasets["train"],  # Dataset de entrenamiento tokenizado
    eval_dataset=tokenized_datasets["test"],    # Dataset de evaluación tokenizado
)
#%%

#%%
# 6. Entrenar el modelo
torch.cuda.empty_cache()
trainer.train()
#%%

# 7. Guardar el modelo fine-tuned
model.save_pretrained("./fine_tuned_chameleon")
