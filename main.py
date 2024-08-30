#%%
import pandas as pd
from transformers import Trainer, TrainingArguments, ChameleonProcessor, ChameleonForConditionalGeneration
from app.config.config import TOKEN_HUGGINGFACE, MODEL_NAME, get_device, print_device_info, LEARNING_RATE, MAX_STEPS_PER_EPOCH, EPOCHS
from app.data.dataset import CustomDataset
from app.utils import hf_login
from app.services.train import train_model
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
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
    # Tokenizar el texto y usar los input_ids como etiquetas
    tokenized_inputs = processor(concatenated_texts, padding="max_length", truncation=True, max_length=1)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()  # Usar clone() en lugar de copy()
    return tokenized_inputs

# Tokenizar el dataset dividido (no el original)
tokenized_datasets = tokenized_datasets.map(tokenize_function, batched=True)
print(tokenized_datasets)

# Cargar los datos con DataLoader y collate_fn para manejar el batching
selected_indices = list(range(1))  # Seleccionar los índices del 0 al 9
train_loader = DataLoader(tokenized_datasets["train"].select(selected_indices), batch_size=1, shuffle=True)
#%%


#%%
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
accumulation_steps = 4  # Número de pasos para acumular gradientes
model.gradient_checkpointing_enable()  # Activar gradient checkpointing

# 6. Entrenar el modelo
model.train()
for epoch in range(5):
    for i, batch in enumerate(train_loader):
        print(batch)  # Imprime la estructura del batch
        # Convertir listas de tensores en un solo tensor y mover a GPU
        # Verifica que los tensores tengan las dimensiones correctas
        input_ids = torch.stack(batch["input_ids"]).to(device).permute(1, 0)
        attention_mask = torch.stack(batch["attention_mask"]).to(device).permute(1, 0)
        labels = torch.stack(batch["labels"]).to(device).permute(1, 0)

        # Verifica dimensiones de los tensores
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }        

        optimizer.zero_grad()  # Resetear los gradientes antes de cada paso de optimización
        outputs = model(**batch)  # Hacer la predicción
        print(outputs)  # Imprimir los resultados
        loss = outputs.loss  # Calcular la pérdida
        loss.backward()  # Propagar los gradientes
        print(f"Loss: {loss}")

        if(i>=8): #(i + 1) % accumulation_steps == 0:  # Actualizar cada n pasos
            print(f"Step {i + 1} - Updating parameters")
            with torch.no_grad():
                optimizer.step()  # Actualizar los parámetros del modelo
            optimizer.zero_grad()  # Resetear los gradientes
            torch.cuda.empty_cache()  # Liberar memoria
        
        # Imprimir la pérdida para monitorear el entrenamiento
        print(f"Loss: {loss.item()}")
        
        # Opcional: liberar memoria no utilizada en GPU
        torch.cuda.empty_cache()

#%%
test_prompt = "Como estas Felipe, que tal tus vacaciones en la playa?"
test_inputs = processor(test_prompt, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
output = model.generate(**test_inputs, max_new_tokens=50)
print(processor.decode(output[0], skip_special_tokens=True))
#%%
# 7. Guardar el modelo fine-tuned
model.save_pretrained("./fine_tuned_chameleon")
#%%