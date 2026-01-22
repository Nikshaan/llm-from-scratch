import re
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

if __name__ == "__main__":
    with open('data/the-verdict.txt', 'r', encoding='utf-8') as f:
        raw_text = f.read()

    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]

    all_words = sorted(set(preprocessed))
    vocab_size = len(all_words)
    # print(f'Vocabulary size: {vocab_size}') # story's vocabulary range

    vocab = {token: integer for integer, token in enumerate(all_words)} # providing ids to each token (like indexing)
    for i, item in enumerate(vocab.items()):
        # print(item)
        if i >= 50:
            break
    
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
        
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed] # tokens to ids
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s([,.:;?_!"()\'])', r'\1', text) # makes 'hello , world' into 'hello, world'
        return text # ids to tokens
    
if __name__ == "__main__":
    all_tokens = sorted(list(set(preprocessed)))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token: integer for integer, token in enumerate(all_tokens)}

    for i, item in enumerate(list(vocab.items())[-5::]):
        # print(item)
        pass

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
        
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed] # replace unknown tokens with <|unk|>
        ids = [self.str_to_int[s] for s in preprocessed] # tokens to ids
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s([,.:;?_!"()\'])', r'\1', text) # makes 'hello , world' into 'hello, world'
        return text # ids to tokens
    
if __name__ == "__main__":
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = ' <|endoftext|> '.join([text1, text2])
    # print(text)
    # tokenizer = SimpleTokenizerV2(vocab)
    # print(tokenizer.encode(text))
    # print(tokenizer.decode(tokenizer.encode(text)))


if __name__ == "__main__":
    # using byte pair encding (BPE)
    text = 'Akwirw ier'
    tokenizer = tiktoken.get_encoding("gpt2")
    integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    # print(integers) # breaks down unknown words into subwords, assigning ids from gpt2 vocab for each subword
    strings = tokenizer.decode(integers)
    # print(strings)

    enc_text = tokenizer.encode(raw_text)
    # print(len(enc_text))
    enc_sample = enc_text[:50]
    # print(enc_sample)

    context_size = 4
    x = enc_sample[:context_size] # current token list
    y = enc_sample[1:context_size+1] # target token list (next token for each in x)
    # print(f"x: {x}")
    # print(f"y: {y}")

    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        # print(f"Context: {context} -> Desired: {desired}")
        # print(tokenizer.decode(context), "->", tokenizer.decode([desired]))

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(txt)
        
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype = torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype = torch.long))
            
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader

if __name__ == "__main__":
    dataloader = create_dataloader_v1(
        raw_text, batch_size = 2, max_length = 8, stride = 4,
        shuffle = False
    )

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    # print("Inputs:\n", inputs)
    # print("Targets:\n", targets)

    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    max_length = 4
    dataloader = create_dataloader_v1(
        raw_text, batch_size = 8, max_length = max_length, stride = max_length, shuffle=False
    )

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    # print("token ids:\n", inputs)
    # print("\n input shape:", inputs.shape)

    token_embeddings = token_embedding_layer(inputs)
    # print(token_embeddings.shape)

    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    # print("positional embeddings shape:", pos_embeddings.shape)

    input_embeddings = token_embeddings + pos_embeddings
    # print("input embeddings shape:", input_embeddings.shape)
