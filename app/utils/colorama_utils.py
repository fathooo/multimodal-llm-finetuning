from colorama import Fore, Style, init

def initialize_colorama():
    init(autoreset=True)

def print_message(message, color=Fore.GREEN):
    print(color + message + Style.RESET_ALL)