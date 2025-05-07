import os
from pathlib import Path
from urllib.parse import urlparse
import opendatasets as od

def download_data(data_URL: str):

    parsed_url = urlparse(data_URL)
    path = parsed_url.path
    data_dir = path.split('/')[-1]
    image_path = Path(f"{data_dir}/")

    if image_path.is_dir():
        print(f"{image_path} already exists.")
    else:
        print("Downloading data...")
        od.download(data_URL)
    
    train_dir = image_path/ "train"
    test_dir = image_path/ "test"

    return train_dir, test_dir