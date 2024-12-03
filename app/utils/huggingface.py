from huggingface_hub import login

def hf_login(token):
    login(token, add_to_git_credential=True)