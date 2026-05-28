import os
import sys
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.json"
]

MODEL_DIR = "models/Qwen2.5-1.5B-Instruct"
os.makedirs(MODEL_DIR, exist_ok=True)
BASE_URL = "https://modelscope.cn/api/v1/models/qwen/Qwen2.5-1.5B-Instruct/repo?Revision=master&FilePath="

def download_file_sequential(filename):
    dest_path = os.path.join(MODEL_DIR, filename)
    url = BASE_URL + filename
    print(f"Downloading {filename}...")
    response = requests.get(url)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(response.content)

def download_chunk(url, start, end, dest_path, progress_callback):
    headers = {"Range": f"bytes={start}-{end}"}
    # Retry logic
    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()
            with open(dest_path, "r+b") as f:
                f.seek(start)
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
                        progress_callback(len(chunk))
            return True
        except Exception as e:
            if attempt == 2:
                print(f"\nChunk {start}-{end} failed after 3 attempts: {e}")
                raise e

def download_large_file_parallel(filename, num_threads=8):
    dest_path = os.path.join(MODEL_DIR, filename)
    url = BASE_URL + filename
    
    # Get total size
    response = requests.head(url)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    print(f"Downloading {filename} ({total_size / (1024*1024*1024):.2f} GB) in parallel using {num_threads} threads...")
    
    # Pre-allocate file space
    with open(dest_path, "wb") as f:
        f.truncate(total_size)
        
    chunk_size = total_size // num_threads
    futures = []
    
    progress = tqdm(total=total_size, unit='B', unit_scale=True, desc=filename)
    
    def update_progress(n):
        progress.update(n)
        
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(num_threads):
            start = i * chunk_size
            end = total_size - 1 if i == num_threads - 1 else (i + 1) * chunk_size - 1
            futures.append(executor.submit(download_chunk, url, start, end, dest_path, update_progress))
            
        for future in as_completed(futures):
            future.result() # Raise exception if any thread failed
            
    progress.close()
    print(f"Finished downloading {filename} in parallel.")

def main():
    print("Downloading small config and tokenizer files sequentially...")
    for filename in FILES:
        download_file_sequential(filename)
        
    print("\nStarting parallel download for model weights...")
    download_large_file_parallel("model.safetensors", num_threads=8)
    print("\nAll downloads completed successfully!")

if __name__ == "__main__":
    main()
