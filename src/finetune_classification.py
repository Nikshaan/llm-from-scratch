import urllib.request
import zipfile
import os
from pathlib import Path
import pandas as pd
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from train import GPT_CONFIG_124M, GPTModel, generate_text_simple, text_to_token_ids, token_ids_to_text
import time
import matplotlib.pyplot as plt

url = "http://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

zip_path = "sms_spam_collection.zip"

extracted_path = "sms_spam_collection"

data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"Data file already exists at {data_file_path}. Skipping download.")
        return

    with urllib.request.urlopen(url) as response, open(zip_path, 'wb') as out_file:
        out_file.write(response.read())

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_path)

    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"Data file extracted to {data_file_path}.")
    
download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

df = pd.read_csv(data_file_path, sep='\t', header=None, names=['Label', 'Text'])

# print(df["Label"].value_counts())

# equalize the number of 'ham' and 'spam' entries

def create_balanced_dataset(df):
    num_spam = df[df['Label'] == 'spam'].shape[0]
    ham_subset = df[df['Label'] == 'ham'].sample(n = num_spam, random_state = 123)
    balanced_df = pd.concat([ham_subset, df[df['Label'] == 'spam']])
    return balanced_df

balanced_df = create_balanced_dataset(df)
# print(balanced_df["Label"].value_counts())

# split the balanced dataset into train (70%), validation (10%), and test (20%) sets

def random_split(df, train_frac, validation_frac):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)  # Shuffle the DataFrame
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df

train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

train_df.to_csv("train.csv", index=False)
validation_df.to_csv("validation.csv", index=False)
test_df.to_csv("test.csv", index=False)

# creating data loaders

tokenizer = tiktoken.get_encoding("gpt2")
# print(tokenizer.encode('<|endoftext|>', allowed_special = {"<|endoftext|>"}))  # [50256]
# we use this as padding token for classification task

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length = None, pad_token_id = 50256):
        self.data = pd.read_csv(csv_file)
   
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data['Text']
        ]
        
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            
        # Truncate or pad all texts to max_length
        self.encoded_texts = [
            encoded_text[:self.max_length] for encoded_text in self.encoded_texts
        ]
        
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text)) 
            for encoded_text in self.encoded_texts
        ]
            
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]['Label']
        # Convert string labels to integers: 'spam' -> 1, 'ham' -> 0
        label = 1 if label == 'spam' else 0
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype = torch.long)
        )
        
    def __len__(self):
        return len(self.data)
    
    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length

train_dataset = SpamDataset(csv_file = "train.csv", max_length = None, tokenizer = tokenizer)
# print(train_dataset.max_length)

val_dataset = SpamDataset(csv_file = "validation.csv", max_length = train_dataset.max_length, tokenizer = tokenizer)

test_dataset = SpamDataset(csv_file = "test.csv", max_length = train_dataset.max_length, tokenizer = tokenizer)

num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers = num_workers,
    drop_last = True
)

val_loader = DataLoader(
    dataset = val_dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers = num_workers,
    drop_last = True
)

test_loader = DataLoader(
    dataset = test_dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers = num_workers,
    drop_last = True
)

for input_batch, target_batch in train_loader:
    pass

# print("Input batch dimensions: ", input_batch.shape)
# print("Label batch dimensions: ", target_batch.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "dropout": 0.1,
    "qkv_bias": True,
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_heads": 12, "n_layers": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_heads": 16, "n_layers": 24},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_heads": 20, "n_layers": 36},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_heads": 25, "n_layers": 48}
}

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
base_model = GPTModel(BASE_CONFIG)

checkpoint = torch.load("gpt2-small-124M.pth", weights_only=True, map_location="cpu")

if "model_state_dict" in checkpoint:
    base_model.load_state_dict(checkpoint["model_state_dict"])
else:
    base_model.load_state_dict(checkpoint)

base_model.to(device) # Move model to GPU/CPU
base_model.eval() # Set to evaluation mode

tokenizer = tiktoken.get_encoding("gpt2")

'''
text_1 = "Every effort moves you"
token_ids = generate_text_simple(
    model=base_model,
    idx=text_to_token_ids(text_1, tokenizer).to(device),
    max_new_tokens=15,
    context_size=1024
)

# print(token_ids_to_text(token_ids, tokenizer))

text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no': "
    "'You are a winner you have been specially selected to receive $1000 cash or a $2000 award.'"
)

token_ids = generate_text_simple(
    model=base_model,
    idx=text_to_token_ids(text_2, tokenizer).to(device),
    max_new_tokens=23,
    context_size=1024
)
# print(token_ids_to_text(token_ids, tokenizer))
'''

# print(base_model)

'''
Model architecture:

GPTModel(
    (tok_emb): Embedding(50257, 768)
    (pos_emb): Embedding(1024, 768)
    (drop_emb): Dropout(p=0.1, inplace=False)
    (trf_blocks): Sequential(
        (ll): TransformerBlock(
            (att): MultiHeadAttention(
                (W-query): Linear(in_features=768, out_features=768, bias=True)
                (W-key): Linear(in_features=768, out_features=768, bias=True)
                (W-value): Linear(in_features=768, out_features=768, bias=True)
                (out_proj): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
            )
            (ff): FeedForward(
                (layers): Sequential(
                    (0): Linear(in_features=768, out_features=3072, bias=True)
                    (1): GELU()
                    (2): Linear(in_features=3072, out_features=768, bias=True)
                )
            )
        (nomr1): LayerNorm()
        (norm2): LayerNorm()
        (drop_resid): Dropout(p=0.0, inplace=False)
        )
    )
    (final_norm): LayerNorm()
    (out_head): Linear(in_features=768, out_features=50257, bias=False)
)
'''

for param in base_model.parameters():
    param.requires_grad = False

# replace base_model.out_head with a new classification head

torch.manual_seed(123)
num_classes = 2  # spam or ham
base_model.out_head = torch.nn.Linear(BASE_CONFIG["emb_dim"], num_classes)
base_model.to(device)

for param in base_model.trf_blocks[-1].parameters():
    param.requires_grad = True
    
for param in base_model.final_norm.parameters():
    param.requires_grad = True

# base_model's out_head, last transformer block, and final normalization layer norm are trainable to fine-tune for spam classification

inputs = tokenizer.encode("Do you have time")
inputs = torch.tensor(inputs).unsqueeze(0)  # add batch dimension
# print("Inputs:", inputs)
inputs = inputs.to(device)
# print("Input shape:", inputs.shape)

with torch.no_grad():
    outputs = base_model(inputs)
# print("Outputs:", outputs)
# print("Outputs shape:", outputs.shape)

# we will focus on the last row corresponding to the last token since it contains the aggregate information for classification

# print("Last output token: ", outputs[:, -1, :])

probas = torch.softmax(outputs[:, -1, :], dim=-1)
label = torch.argmax(probas, dim=-1)
# print("Predicted label: ", label.item())  # 0 for 'ham', 1 for 'spam'

# simplify without using softmax

logits = outputs[:, -1, :]
label = torch.argmax(logits, dim=-1)
# print("Predicted label (without softmax): ", label.item())

def calc_accuracy_loader(data_loader, model, device, num_batches = None):
    model.eval()
    correct_predictions, num_examples = 0, 0
    
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
        
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            with torch.no_grad():
                outputs = model(input_batch)
                
            logits = outputs[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)
            
            num_examples += target_batch.size(0)
            correct_predictions += (predicted_labels == target_batch).sum().item()
            
        else:
            break
    return correct_predictions / num_examples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model.to(device)

torch.manual_seed(123)
train_accuracy = calc_accuracy_loader(train_loader, base_model, device, num_batches = 10)
val_accuracy = calc_accuracy_loader(val_loader, base_model, device, num_batches = 10)
test_accuracy = calc_accuracy_loader(test_loader, base_model, device, num_batches = 10)

# print(f"Initial Train Accuracy: {train_accuracy * 100:.2f}%")
# print(f"Initial Validation Accuracy: {val_accuracy * 100:.2f}%")
# print(f"Initial Test Accuracy: {test_accuracy * 100:.2f}%")

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits, target_batch)
    
    return loss

def calc_loss_loader(data_loader, model, device, num_batches = None):
    model.eval()
    total_loss = 0.0
    num_examples = 0
    
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
        
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            batch_size = input_batch.size(0)
            total_loss += loss.item() * batch_size
            num_examples += batch_size
        else:
            break
    return total_loss / num_examples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model.to(device)   

torch.manual_seed(123)
train_loss = calc_loss_loader(train_loader, base_model, device, num_batches = 10)
val_loss = calc_loss_loader(val_loader, base_model, device, num_batches = 10)
test_loss = calc_loss_loader(test_loader, base_model, device, num_batches = 10)

# print(f"Initial Train Loss: {train_loss:.4f}")
# print(f"Initial Validation Loss: {val_loss:.4f}")
# print(f"Initial Test Loss: {test_loss:.4f}")

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches = eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches = eval_iter)
    model.train()
    return train_loss, val_loss

def train_classifier_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []  
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):     
        model.train()                   

        for input_batch, target_batch in train_loader:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            optimizer.zero_grad()               
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()                     
            optimizer.step()                    
            examples_seen += input_batch.shape[0]   
            global_step += 1

            if global_step % eval_freq == 0:    
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}")

        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)

        print(f"Training accuracy: {train_accuracy*100:.2f}%")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen

start_time = time.time()
torch.manual_seed(123)

optimizer = torch.optim.AdamW(base_model.parameters(), lr = 5e-5, weight_decay = 0.1)
num_epochs = 5

train_losses, val_losses, train_accs, val_accs, examples_seen = \
    train_classifier_simple(
        base_model, train_loader, val_loader, optimizer, device,
        num_epochs = num_epochs, eval_freq = 50,
        eval_iter = 5
    )

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes")

def plot_values(
    epochs_seen, examples_seen, train_values, val_values,
    label = "loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs_seen, train_values, label = f"Training {label}")
    ax1.plot(
        epochs_seen, val_values, linestyle = "-.",
        label = f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha = 0)
    ax2.set_xlabel("Examples seen")
    fig.tight_layout()
    plt.show()

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

# no overfitting as no noticible gap between training and validation loss/accuracy

# plot classification accuracies

epochs_tensor = torch.linspace(1, num_epochs, len(train_accs))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))

plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label = "accuracy")

# training and validation accuracies increase over epochs

# check training, validation and test accuracies after fine-tuning

train_accuracy = calc_accuracy_loader(train_loader, base_model, device)
val_accuracy = calc_accuracy_loader(val_loader, base_model, device)
test_accuracy = calc_accuracy_loader(test_loader, base_model, device)

print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")

def classify_review(
    text, model, tokenizer, device, max_length=None,
    pad_token_id=50256):
    model.eval()

    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]

    input_ids = input_ids[:min(max_length, supported_context_length)]

    input_ids += [pad_token_id] * (max_length - len(input_ids))

    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"

text_1 = ("You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.")

print(classify_review(text_1, base_model, tokenizer, device, max_length = train_dataset.max_length))

text_2 = ("Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!")

print(classify_review(text_2, base_model, tokenizer, device, max_length = train_dataset.max_length))

torch.save(base_model.state_dict(), "review_classifier.pth")
model_state_dict = torch.load("review_classifier.pth", map_location=device)
base_model.load_state_dict(model_state_dict)