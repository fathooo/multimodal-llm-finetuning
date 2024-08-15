from huggingface_hub import login

def hf_login(token):
    login(token)