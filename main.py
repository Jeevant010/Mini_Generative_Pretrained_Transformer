import torch

chars = ""
with open('../wizard_of_oz.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))
    
    
string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}

# Encode: string -> list of integers
encode = lambda s: [string_to_int[c] for c in s]

# Decode: list of integers -> string
decode = lambda l: ''.join(int_to_string[i] for i in l)

# Convert encoded data to a tensor
data = torch.tensor(encode(text), dtype=torch.long)


prompt = "Hello! Where the hell are you?"
context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
generated_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens=100)[0].tolist())
print(generated_chars)