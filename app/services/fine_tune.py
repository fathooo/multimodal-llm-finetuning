import torch
import os

def fine_tune_model(model, train_loader, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    accumulation_steps = 4  # Número de pasos para acumular gradientes
    model.gradient_checkpointing_enable()  # Activar gradient checkpointing

    model.train()
    for epoch in range(1):
        for i, batch in enumerate(train_loader):
            print(batch)  # Imprime la estructura del batch

            # Convertir listas de tensores en un solo tensor y mover a GPU
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

            if (i >= 8):  # (i + 1) % accumulation_steps == 0:  # Actualizar cada n pasos
                print(f"Step {i + 1} - Updating parameters")
                with torch.no_grad():
                    optimizer.step()  # Actualizar los parámetros del modelo
                optimizer.zero_grad()  # Resetear los gradientes
                torch.cuda.empty_cache()  # Liberar memoria

            # Imprimir la pérdida para monitorear el entrenamiento
            print(f"Loss: {loss.item()}")

            # Opcional: liberar memoria no utilizada en GPU
            torch.cuda.empty_cache()

def save_model(model, path="./model/fine_tuned_chameleon"):
    # tratar con path
    if not path.endswith("/"):
        path += "/"
    
    if not path.exists():
        os.makedirs(path)
        
    model.save_pretrained(path)