# Google Colab
# Para ejecutar esto en Google Colab, asegúrate de que estás utilizando un entorno con soporte de GPU.
# Instalar librerías necesarias
# !pip install --upgrade torch torchvision transformers datasets pillow requests huggingface_hub

# Importar las librerías necesarias
import torch
from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
from PIL import Image
import pandas as pd
import requests
from huggingface_hub import login
from torch.utils.data import DataLoader, Dataset

# Constantes
TOKEN_HUGGINGFACE = "wed_pasdGsaazcwrjas34IvHOUHNtRasdasda"

# Iniciar sesión en Hugging Face
login(TOKEN_HUGGINGFACE)

# Configuración inicial y carga de modelo
model_name = "facebook/chameleon-7b"
processor = ChameleonProcessor.from_pretrained(model_name)
model = ChameleonForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")

# Functions
## Función para descargar y procesar imágenes y texto
def process_data(row):
    response = requests.get(row['image_url'], stream=True)
    image = Image.open(response.raw)
    inputs = processor(row["text"] + "<image>", images=image, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
    return inputs

# Creación de un mock dataset
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
df = pd.DataFrame(data)

# Mostrar el dataset
print(df.head())

# Probar con un solo dato
sample_data = process_data(df.iloc[0])
print(sample_data)




class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return process_data(self.data.iloc[idx])

# Crear DataLoader
train_dataset = CustomDataset(df)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

#%%
# Entrenamiento
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Loop de entrenamiento
model.train()
for epoch in range(3):  # Número de épocas de ejemplo
    for batch in train_dataloader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        print(f"Loss: {loss.item()}")  # Imprimir pérdida
        loss.backward()
        optimizer.step()


#%%
# Guardar el modelo entrenado
model.save_pretrained("./save-finetuned-model")
processor.save_pretrained("./save-finetuned-processor")

