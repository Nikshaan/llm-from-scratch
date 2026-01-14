import torch
from model import GPTModel
from model import generate_text_simple
import tiktoken
from data import create_dataloader_v1
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import urllib.request

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "emb_dim": 768,
    "context_length": 256,
    "n_layers": 12,
    "n_heads": 12,
    "dropout": 0.1,
    "qkv_bias": False,
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.flatten()
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model = model,
    idx = text_to_token_ids(start_context, tokenizer),
    max_new_tokens = 10,
    context_size = GPT_CONFIG_124M["context_length"]
)

# print("Output text: ", token_ids_to_text(token_ids, tokenizer))

inputs = torch.tensor([[16833, 3626, 6100],
                      [40, 1107, 588]])

targets = torch.tensor([[3626, 6100, 345],
                        [1107, 588, 11311] ])

with torch.no_grad():
    logits = model(inputs)
probas = torch.softmax(logits, dim=-1)
# print(probas.shape)

token_ids = torch.argmax(probas, dim=-1, keepdim=True)
# print("Token IDs:\n", token_ids)

# print("Targets batch 1: ", (token_ids_to_text(targets[0], tokenizer)))
# print("Predicted batch 1: ", (token_ids_to_text(token_ids[0], tokenizer)))

text_idx = 0
target_probas_1 = probas[text_idx, [0,1,2], targets[text_idx]]
# print("Text 1: ", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0,1,2], targets[text_idx]]
# print("Text 2: ", target_probas_2)
#  three target probabilities for each text in the batch

log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
# print("Log probabilities: ", log_probas)

avg_log_proba = torch.mean(log_probas)
# print("Average log probability: ", avg_log_proba) # need to get this close to 0

# cross entropy loss is used as it measures the difference between two probability distributions (true & predicted) (negative avg log proba)

# print("Logits shape: ", logits.shape)  # (batch_size, seq_length, vocab_size)
# print("Targets shape: ", targets.shape)  # (batch_size, seq_length)

logits_flat = logits.flatten(0, 1)  # (batch_size * seq_length, vocab_size)
targets_flat = targets.flatten() # (batch_size * seq_length)

# print("Logits flat shape: ", logits_flat.shape)
# print("Targets flat shape: ", targets_flat.shape)

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
# print("Cross-entropy loss: ", loss)  # want to minimize this loss

# perplexity - measures how well a probability distribution was predicted by the model
perplexity = torch.exp(loss)
# print("Perplexity: ", perplexity)

with open('data/the-verdict.txt', "r", encoding="utf-8") as f:
    text = f.read()

total_characters = len(text)
total_tokens = len(tokenizer.encode(text, allowed_special={'<|endoftext|>'}))
# print(f"Total characters in text: {total_characters}")
# print(f"Total tokens in text: {total_tokens}")

train_ratio = 0.90
split_idx = int(len(text) * train_ratio)
train_data = text[:split_idx]
val_data = text[split_idx:]

torch.manual_seed(123)
train_loader = create_dataloader_v1(
    train_data,
    batch_size = 2,
    max_length = GPT_CONFIG_124M["context_length"],
    stride = GPT_CONFIG_124M["context_length"],
    shuffle = True,
    drop_last = True,
    num_workers = 0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size = 2,
    max_length = GPT_CONFIG_124M["context_length"],
    stride = GPT_CONFIG_124M["context_length"],
    shuffle = False,
    drop_last = False,
    num_workers = 0
)

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches = None):
    total_loss = 0.
    
    if len(data_loader) == 0:
        return float('nan')
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader)) # data loader is the length of the dataset divided by batch size
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches # average loss over all batches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)
    
# print(f"Train loss: {train_loss}")
# print(f"Validation loss: {val_loss}")


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer): # eval_iter is number of random batches to eval on whereas eval_freq is number of steps between evals
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    
    for epoch in range(num_epochs):
        model.train() # set model to training mode
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # zero the gradients
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # backpropagate the loss
            optimizer.step() # update the model parameters
            tokens_seen += input_batch.numel() # number of elements in the input batch
            global_step += 1
            
            if global_step % eval_freq == 0: # if number of steps == eval frequency
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter) # eval iter here is num batches to eval on
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                
                print(f"Epoch {epoch+1}, Step {global_step}: Train Loss = {train_loss}, Val Loss = {val_loss}")
                
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval() # set model to evaluation mode, dropout disabled
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    model.train() # set model back to training mode
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context): # sanity check to see if model is learning
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model = model,
            idx = encoded,
            max_new_tokens = 50,
            context_size = context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print("Generated text sample:\n", decoded_text.replace('\n', ' '))
    model.train()
    
# using AdamW optimizer minimizes model complexity and prevents overfitting by penalizing large weights

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0004, weight_decay = 0.1)
num_epochs = 10
''' 
train_losses, val_losses, tokens_seen = train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs = num_epochs,
    eval_freq = 5, # eval every 5 steps
    eval_iter = 5, # eval on 5 random batches
    start_context = "Every effort moves you",
    tokenizer = tokenizer
)
'''

# .parameters() method returns all trainable weight parameters of the model


def plot_losses(epochs_seen, token_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(epochs_seen, train_losses, label='Train Loss')
    ax1.plot(epochs_seen, val_losses, label='Validation Loss', color='orange')
    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    ax2.plot(token_seen, train_losses, alpha=0)
    ax2.set_xlabel('Tokens Seen')

    fig.tight_layout()
    plt.show()
'''
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
'''

model.to("cpu")
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
    model = model,
    idx = text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens = 25,
    context_size = GPT_CONFIG_124M["context_length"]
)

# print("Output text: ", token_ids_to_text(token_ids, tokenizer))


# temperature scaling

vocab = {
    "closer": 0,
    "every": 1,
    "effort": 2,
    "forward": 3,
    "inches": 4,
    "moves": 5,
    "pizza": 6,
    "toward": 7,
    "you": 8
}

inverse_vocab = {v: k for k, v in vocab.items()}

next_token_logits = torch.tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])
probas = torch.softmax(next_token_logits, dim=-1)
next_token_id = torch.argmax(probas).item()
# print(inverse_vocab[next_token_id]) # forward

def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item() for _ in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"Token: {inverse_vocab[i]}, Frequency: {freq.item()}")
# print_sampled_tokens(probas)

# multinomial sampling generates diverse outputs by sampling from the probability distribution of next tokens, rather than always choosing the most probable token
# argmax gives deterministic output while multinomial sampling gives diverse outputs

def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=-1)

# temperature > 1 makes distribution more uniform (more exploration)
# temperature < 1 makes distribution peakier (more exploitation)

'''
temperatures = [1, 0.1, 5]                                                      #1
scaled_probas = [softmax_with_temperature(next_token_logits, T)
                 for T in temperatures]
x = torch.arange(len(vocab))
bar_width = 0.15
fig, ax = plt.subplots(figsize=(5, 3))
for i, T in enumerate(temperatures):
    rects = ax.bar(x + i * bar_width, scaled_probas[i], bar_width, label=f'Temperature = {T}')

ax.set_ylabel('Probability')
ax.set_xticks(x)
ax.set_xticklabels(vocab.keys(), rotation=90)
ax.legend()
plt.tight_layout()
plt.show()
'''

# top - k sampling

# restrict the sampling to the top k most probable tokens

top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)
# print("Top logits: ", top_logits)
# print("Top positions: ", top_pos)

# set lowest logit values to -inf so that their softmax probabilities become 0
new_logits = torch.where(condition = next_token_logits < top_logits[-1],
                         input = torch.tensor(float('-inf')),
                         other = next_token_logits)

topk_probas = torch.softmax(new_logits, dim=-1)
# print("Top-k probabilities: ", topk_probas)

def generate(model, idx, max_new_tokens, context_size, temperature = 0.0, top_k = None, eos_id = None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val,
                                 torch.tensor(float('-inf')).to(logits.device),
                                 logits)
            
        if temperature > 0.0:
            logits = logits / temperature
            probas = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probas, num_samples=1)
            
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        
        if idx_next == eos_id:
            break
        
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
            
torch.manual_seed(123)
token_ids = generate(
    model = model,
    idx = text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens = 15,
    context_size = GPT_CONFIG_124M["context_length"],
    top_k = 25,
    temperature = 1.4
)

# print("Output text:", token_ids_to_text(token_ids, tokenizer))

# saving model

torch.save(model.state_dict(), "model.pth")

# loading model

model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# save model and optimizer state dicts together for resuming training

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, "model_optimizer.pth"
)

# restore model and optimizer state dicts

checkpoint = torch.load("model_optimizer.pth")
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.5e-4, weight_decay = 0.1)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.train()

# loading pretrained weights from OpenAI

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "dropout": 0.1,
    "qkv_bias": True
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTModel(GPT_CONFIG_124M)
model.eval()
model.to(device)

state_dict = torch.load("gpt2-small-124M.pth", map_location=device, weights_only=True)
model.load_state_dict(state_dict, strict=True)
tokenizer = tiktoken.get_encoding("gpt2")
start_context = "Every effort moves you"
token_ids = tokenizer.encode(start_context)
token_tensor = torch.tensor(token_ids).unsqueeze(0).to(device)

out = generate(
    model = model,
    idx = token_tensor,
    max_new_tokens=25,
    context_size=1024
)

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
# print(f"Output text:{decoded_text}")
