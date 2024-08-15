#%%

import pandas as pd
from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
from app.config.config import TOKEN_HUGGINGFACE, MODEL_NAME, get_device, print_device_info, LEARNING_RATE, MAX_STEPS_PER_EPOCH, EPOCHS
from app.data.dataset import CustomDataset
from app.utils import hf_login
from app.repository.useCases.train import train_model
import torch

print_device_info()
device = get_device()
print(f"Using device: {device}")

# Iniciar sesión en Hugging Face
hf_login(TOKEN_HUGGINGFACE)

# Configuración inicial y carga de modelo
processor = ChameleonProcessor.from_pretrained(MODEL_NAME)
model = ChameleonForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device)
#%% data
data = {
    "text": [
        "Un perro corriendo en el parque.",
        "Un gato saltando desde un árbol.",
        "Un niño jugando con una pelota.",
        "Un coche rojo estacionado en la calle.",
        "Una hermosa puesta de sol.",
        "Una taza de café en una mesa.",
        "Un grupo de personas en una fiesta."
    ],
    "image_url": [
        "https://cdn.pixabay.com/photo/2022/09/11/15/06/dog-7447075_1280.jpg",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS11VtfGtvMCf5fk33z6SSMk0KLUdtg1_OG7g&s",
        "https://st2.depositphotos.com/2751239/7710/i/450/depositphotos_77108351-stock-photo-two-cute-little-kids-playing.jpg",
        "https://img.freepik.com/fotos-premium/coche-rojo-esta-estacionado-calle-frente-edificio_1089043-92402.jpg",
        "https://i.pinimg.com/236x/d4/8a/ea/d48aea840f5de4bcf2eb4325607b277f.jpg",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS6fjlbTD3cmnHvy1KqXZD5b19_9nhWuEsTxg&s",
        "https://img.freepik.com/fotos-premium/grupo-personas-fiesta-gorro-fiesta_670382-21629.jpg",
    ]
}

# Convertir la data en un DataFrame
df = pd.DataFrame(data)

# Crear un dataset y entrenar el modelo
train_dataset = CustomDataset(df, processor, device)
train_model(model, train_dataset, processor, device, LEARNING_RATE, MAX_STEPS_PER_EPOCH, EPOCHS)
