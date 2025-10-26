import os
import lzma
import tqdm
import concurrent.futures
import random
from datasets import load_dataset

def process_files(args):
    directory, filename, output_file = args
    file_path = os.path.join(directory, filename)
    
    with lzma.open(file_path, "rt", encoding='utf-8') as infile:
        text = infile.read()
    
    with open(output_file, "a", encoding='utf-8') as outfile:
        outfile.write(text)
        
    characters = set(text)
    return characters

def xz_file_in_dir(directory):
    return [
        filename for filename in os.listdir(directory)
        if filename.endswith('.xz') and os.path.isfile(os.path.join(directory, filename))
    ]

def process_file_in_parallel(files, folder_path, output_file):
    vocab = set()
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        args = [(folder_path, filename, output_file) for filename in files]
        for characters in tqdm.tqdm(executor.map(process_files, args), total=len(files)):
            vocab.update(characters)
    return vocab

folder_path = "openwebtext"
output_file_train = "output_train.txt"
output_file_val = "output_val.txt"
vocab_file = "vocab.txt"

files = xz_file_in_dir(folder_path)
total_files = len(files)

split_index = int(total_files * 0.9)
files_train = files[:split_index]
files_val = files[split_index:]

sampling_rate = 0.01
files_train_sample = random.sample(files_train, max(1, int(len(files_train) * sampling_rate)))
files_val_sample = random.sample(files_val, max(1, int(len(files_val) * sampling_rate)))

open(output_file_train, 'w').close()
open(output_file_val, 'w').close()

vocab_train = process_file_in_parallel(files_train_sample, folder_path, output_file_train)
vocab_val = process_file_in_parallel(files_val_sample, folder_path, output_file_val)

vocab = vocab_train.union(vocab_val)
with open(vocab_file, "w", encoding="utf-8") as vfile:
    for char in sorted(vocab):
        vfile.write(char + '\n')
