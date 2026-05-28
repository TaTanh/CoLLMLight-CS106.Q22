import os
import requests
from tqdm import tqdm

FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.json"
]

MODEL_DIR = "models/Qwen2.5-1.5B-Instruct"
os.makedirs(MODEL_DIR, exist_ok=True)

BASE_URL = "https://modelscope.cn/api/v1/models/qwen/Qwen2.5-1.5B-Instruct/repo?Revision=master&FilePath="

print(f"Downloading model files to {MODEL_DIR}...")

for filename in FILES:
    dest_path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(dest_path):
        # Quick validation of existing files
        if filename != "model.safetensors" or os.path.getsize(dest_path) > 3000000000:
            print(f"File {filename} already exists, skipping.")
            continue
            
    url = BASE_URL + filename
    print(f"Downloading {filename}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024 # 1MB
    
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(dest_path, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

print("All downloads completed successfully!")
