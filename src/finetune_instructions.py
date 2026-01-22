import json
import os
from pathlib import Path
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial
from model import GPTModel
from train import generate, text_to_token_ids, token_ids_to_text, calc_loss_loader, train_model_simple, plot_losses
import time
from tqdm import tqdm
import re

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR/"data"/"instruction-data.json"

def load_instruction_data(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find file at: {file_path}")
        
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    try:
        data = load_instruction_data(DATA_PATH)
        print(f"Number of entries: {len(data)}") # dictionary containing instruction, input, output
    except Exception as e:
        print(f"Error loading data: {e}")
    
# using Alpaca prompt style for finetuning

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = (f"\n\n### Input:\n{entry['input']}" if entry['input'].strip() else "")

    return instruction_text + input_text

if __name__ == "__main__":
    model_input = format_input(data[50])
    desired_response = f"\n\n### Response:\n{data[50]['output']}"
    # print(model_input + desired_response)

    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.10)
    val_portion = len(data) - train_portion - test_portion

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    # print(f"Train size: {len(train_data)}")
    # print(f"Test size: {len(test_data)}")
    # print(f"Validation size: {len(val_data)}")

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(self.tokenizer.encode(full_text))
            
    def __getitem__(self, index):
        return self.encoded_texts[index]
    
    def __len__(self):
        return len(self.encoded_texts)
    
if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    # print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

def custom_collate_draft_1(batch, pad_token_id = 50256, device = 'cpu'):
    batch_max_length  = max(len(item) + 1 for item in batch) # +1 for extra pad token added later; longest length in batch
    inputs_lst = []
    
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id] 
        
        padded = (new_item + [pad_token_id] * (batch_max_length - len(new_item)))
        
        inputs = torch.tensor(padded[:-1]) # removes extra padded token added earlier
        inputs_lst.append(inputs)
        
    inputs_tensor = torch.stack(inputs_lst).to(device)
    return inputs_tensor

def custom_collate_draft_2(batch, pad_token_id = 50256, device = 'cpu'):
    batch_max_length  = max(len(item) + 1 for item in batch) # +1 for the extra pad token
    inputs_lst = []
    targets_lst = []
    
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]  # Add one pad token for shifting
        padded = (new_item + [pad_token_id] * (batch_max_length - len(new_item)))

        inputs = torch.tensor(padded[:-1]) # removes extra padded token added earlier
        targets = torch.tensor(padded[1:]) # shift by one for target
        
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

def custom_collate_fn(batch, pad_token_id = 50256, ignored_index = -100, allowed_max_length = None, device = 'cpu'):
    batch_max_length  = max(len(item) + 1 for item in batch) # +1 for the extra pad token
    inputs_lst = []
    targets_lst = []
    
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]  # Add one pad token for shifting
        padded = (new_item + [pad_token_id] * (batch_max_length - len(new_item)))

        inputs = torch.tensor(padded[:-1]) # removes extra padded token added earlier
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id # find padding token indices
        indices = torch.where(mask)[0] # get indices of padding tokens and store first padding token index

        if indices.numel() > 1:# if number of padding tokens > 1
            targets[indices[1:]] = ignored_index # set all but first padding token to ignored_index
            
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
            
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device) # converts list of tensors to single tensor matrix
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

if __name__ == "__main__":
    batch = (
        [0, 1, 2, 3, 4],
        [5, 6],
        [7, 8, 9]
    )

    # print(custom_collate_draft_1(batch, device='cpu'))

    # print(custom_collate_draft_2(batch, device='cpu'))

    # print(custom_collate_fn(batch, device='cpu'))

# by default pytorch's cross entropy loss ignores -100 index

# future:
# we should mask the instruction and question tokens by setting them to -100 in target.
# for input, everything is included.
# for target, only the response part is included.
# thus the loss is only computed over the response tokens.

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    customized_collate_fn = partial(
        custom_collate_fn,
        device = device,
        allowed_max_length = 1024
    )

    num_workers = 0
    batch_size = 2

    torch.manual_seed(123)

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        collate_fn = customized_collate_fn,
        num_workers = num_workers,
        shuffle = True,
        drop_last=True
    )

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size = batch_size,
        collate_fn = customized_collate_fn,
        num_workers = num_workers,
        shuffle = False,
        drop_last = False
    )

    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        collate_fn = customized_collate_fn,
        num_workers = num_workers,
        shuffle = False,
        drop_last = False
    )

if __name__ == "__main__":
    # print("TRAIN LOADER:")

    # for inputs, targets in train_loader:
        # print("Inputs shape:", inputs.shape)
        # print("Targets shape:", targets.shape)

    # loading pretrained LLM

    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "dropout": 0.0,
        "qkv_bias": True,
        "emb_dim": 768,
        "n_layers": 12,
        "n_heads": 12
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(BASE_CONFIG)
    model.to(device)

    weights_path = "gpt2-small-124M.pth" 
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)

    model.load_state_dict(state_dict, strict=True) 
    model.eval()

    torch.manual_seed(123)
    input_text = format_input(val_data[0])
    # print(input_text)

    token_ids = generate(
        model,
        idx = text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens = 35,
        context_size = BASE_CONFIG["context_length"],
        eos_id = 50256
    )

    generated_text = token_ids_to_text(token_ids, tokenizer)

    model.to(device)
    torch.manual_seed(123)

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)

    # print("Training loss:", train_loss)
    # print("Validation loss:", val_loss)

    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay = 0.1) # weight decay helps in regularization by penalizing large weights

    num_epochs = 2

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs = num_epochs, eval_freq = 5, eval_iter = 5,
        start_context = format_input(val_data[0]), tokenizer = tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in: {execution_time_minutes:.2f} minutes")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, save_path = "instruction_finetuning_loss.png")

    for entry in test_data[:3]: # iterate over first three test entries
        input_text = format_input(entry)
        token_ids = generate(
            model,
            idx = text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens = 256,
            context_size = BASE_CONFIG["context_length"],
            eos_id = 50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)

        response_text = (generated_text[len(input_text):]).replace('### Response:', '').strip()

        print("INPUT PROMPT:")
        print(input_text)
        print("GENERATED RESPONSE:")
        print(generated_text)
        
    # 50 times bigger dataset: https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json

    for i, entry in tqdm(enumerate(test_data), total = len(test_data)):
        input_text = format_input(entry)
        token_ids = generate(
            model,
            idx = text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens = 256,
            context_size = BASE_CONFIG["context_length"],
            eos_id = 50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)

        response_text = (generated_text[len(input_text):]).replace('### Response:', '').strip()

        test_data[i]["model_response"] = response_text

    with open(BASE_DIR/"data"/"instruction-data-with-responses.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=4)

    file_name = f"{re.sub(r'[ ()]', '', str(BASE_CONFIG))}-sft.pth"
    torch.save(model.state_dict(), BASE_DIR/"models"/file_name)
    print(f"Model weights saved to {BASE_DIR/'models'/file_name}")

# can be loaded by using model.load_state_dict(torch.load(path))
