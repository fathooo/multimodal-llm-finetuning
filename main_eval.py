import os
import torch
from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
from app.config.environments import ENV_DOWNLOAD_DATA_DIR, ENV_DOWNLOAD_DATA_EVAL_DIR
from app.config.config import TOKEN_HUGGINGFACE, MODEL_NAME, get_device, print_device_info, LEARNING_RATE, MAX_STEPS_PER_EPOCH, EPOCHS
from app.data.download import download_datasets
from app.data.preprocess import load_dataset, split_dataset, load_evals_dataset
from app.utils.huggingface import hf_login
from app.services.train import train_model
from app.utils.colorama_utils import initialize_colorama, print_message
from colorama import Fore, Style
from app.config.dataset_info import dataset_info_list, dataset_evals_list
import pandas as pd
import random
import time

print("\n\n=============init=============\n\n")
initialize_colorama()
print_device_info()
device = get_device()
print_message(f"Using device: {device}", Fore.GREEN)

# Login to Hugging Face
hf_login(TOKEN_HUGGINGFACE)

# Download datasets if they don't exist
download_datasets(dataset_evals_list, ENV_DOWNLOAD_DATA_EVAL_DIR)

# Check if the file exists before loading
# Charge the first dataset
file_info = dataset_evals_list[0]
file_path = os.path.join(ENV_DOWNLOAD_DATA_EVAL_DIR, file_info['file_name'])
if not os.path.exists(file_path):
    print_message("File does not exist.", Fore.RED)

# Initialize processor to get tokenizer
processor = ChameleonProcessor.from_pretrained(MODEL_NAME)
tokenizer = processor.tokenizer


def generate_answer(question, options, model, tokenizer, device):
    input_text = f"Pregunta: {question}\nOpciones: {options}"
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_length=50)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Función de prueba para generar respuestas aleatorias
def generate_answer_testing(question, options, model, tokenizer, device):
    possible_answers = ['A', 'B', 'C', 'D']
    random_answer = random.choice(possible_answers)
    return random_answer

# Load local dataset
df = load_evals_dataset(file_path, file_info['format'])
model = None

# model = ChameleonForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device)
# train_model(model, train_dataset, processor, device, LEARNING_RATE, MAX_STEPS_PER_EPOCH, EPOCHS)
# Crear lista para almacenar resultados
    
test_results = []
max_iterations = 100

# Iterar sobre cada fila del dataframe y realizar inferencia de prueba
for idx, row in df.iterrows():
    iter = idx + 1
    if iter > max_iterations:
        break
    question = row['Question']
    options = f"A) {row['A']}\nB) {row['B']}\nC) {row['C']}\nD) {row['D']}"
    correct_answer = row['Answer']
    
    # Medir el tiempo de inicio
    start_time = time.time()
    
    # Usar la función de prueba para generar una respuesta aleatoria
    generated_answer = generate_answer_testing(question, options, model=None, tokenizer=None, device=None)
    
    # Calcular el tiempo de respuesta
    response_time = time.time() - start_time
    
    # Comparar la respuesta generada con la correcta
    is_correct = generated_answer == correct_answer
    
    # Almacenar el resultado en la lista de pruebas
    test_results.append({
        "Question": question,
        "Generated Answer (Testing)": generated_answer,
        "Correct Answer": correct_answer,
        "Is Correct": is_correct,
        "Response Time (s)": response_time
    })
    
    # Imprimir el resultado de la prueba
    print_message(f"Question: {question}", Fore.BLUE)
    print_message(f"Options:\n{options}", Fore.MAGENTA)
    print_message(f"Generated Answer (Testing): {generated_answer}", Fore.YELLOW)
    print_message(f"Correct Answer: {correct_answer}", Fore.GREEN)
    print_message(f"Is Correct: {is_correct}", Fore.CYAN)
    print_message(f"Response Time (s): {response_time:.4f}", Fore.CYAN)
    print(Style.RESET_ALL)

# Convertir resultados de prueba a un DataFrame y guardar en CSV
test_results_df = pd.DataFrame(test_results)
test_results_df.to_csv(f'{ENV_DOWNLOAD_DATA_EVAL_DIR}/evaluation_results_testing.csv', index=False)

print_message("Testing evaluation completed. Results saved to 'evaluation_results_testing.csv'", Fore.GREEN)