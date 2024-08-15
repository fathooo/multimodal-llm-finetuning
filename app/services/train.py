import torch
from torch.utils.data import DataLoader

def train_model(model, train_dataset, processor, device, learning_rate, max_steps_per_epoch, epochs):
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(train_dataloader):
            if step >= max_steps_per_epoch:
                break
            optimizer.zero_grad()
            # Elimina la dimensión extra en 'input_ids', 'attention_mask' y 'pixel_values'
            batch = {k: v.squeeze(1) if k in ['input_ids', 'attention_mask'] and v.dim() > 2 else v for k, v in batch.items()}
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Imprimir formas, tipos y dispositivos de los tensores
            for key, value in batch.items():
                print(f"{key} shape: {value.shape}, dtype: {value.dtype}, device: {value.device}")
            
            # Convertir los inputs a los tensores que el modelo espera
            print(f"Running step {step+1}/{max_steps_per_epoch}")
            outputs = model(**batch)            
            print(f"Model output: {outputs}")
            loss = outputs.loss
            print(f"Step {step+1}/{max_steps_per_epoch}, Loss: {loss.item()}")  # Imprimir pérdida
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} completed")

    print("Training completed")