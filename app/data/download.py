import os
import requests
from colorama import Fore
from app.utils.colorama_utils import print_message

def download_datasets(dataset_info_list, download_dir):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    for dataset_info in dataset_info_list:
        url = dataset_info['url']
        output_path = os.path.join(download_dir, dataset_info['file_name'])

        if not os.path.exists(output_path):
            print_message(f"Downloading {dataset_info['file_name']} from {url}", Fore.CYAN)
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print_message(f"Downloaded {dataset_info['file_name']} to {output_path}", Fore.GREEN)
            except requests.exceptions.RequestException as e:
                print_message(f"Failed to download {dataset_info['file_name']} from {url}: {e}", Fore.RED)
                if os.path.exists(output_path):
                    os.remove(output_path)
        else:
            print_message(f"{dataset_info['file_name']} already exists in {download_dir}", Fore.YELLOW)