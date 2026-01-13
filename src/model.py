import torch
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt
from attention import MultiHeadAttention

GPT_CONFIG_124  = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "dropout": 0.1,
    "qkv_bias": False,
}

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"]) # token embedding
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"]) # position embedding 
        self.drop_emb = nn.Dropout(cfg["dropout"])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg)
              for _ in range(cfg["n_layers"])] # transformer blocks (n_layers times)
        )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"]) # final layer of normalization
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False   # output head to vocab size, where each dimension represents a token likelihood
        )
        
    def forward(self, x):
        batch_size, seq_length = x.shape
        tok_embeds = self.tok_emb(x)
        pos_embeds = self.pos_emb(
            torch.arange(seq_length, device=x.device)   
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    
    def forward(self, x):
        return x    

class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        
    def forward(self, x):
        return x
    
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
# print(batch)

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124)
logits = model(batch) # 2 x 4 x 50257 (each word is a vector of 50257 dimension showing next token likelihood)
# print(logits)
# print(logits.shape)

torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
# print(out)

mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
# print(mean)
# print(var)

out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True) # should be close to 0
var = out_norm.var(dim=-1, keepdim=True) # should be close to 1
torch.set_printoptions(sci_mode=False)
# print(out_norm)
# print(mean)
# print(var)

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False) # unbiased=False for population variance, makes n - 1 denominator n
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    
ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased = False, keepdim=True)
# print(var)
# print(mean)

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
    
gelu, relu = GELU(), nn.ReLU()

x = torch.linspace(-3, 3, steps=100) # 100 data samples from -3 to 3
y_gelu, y_relu = gelu(x), relu(x)
# plt.figure(figsize=(8, 8))
# for i, (y, label) in enumerate(zip([y_gelu, y_relu], ['GELU', 'ReLU'])):
    # plt.subplot(2, 1, i + 1)
    # plt.plot(x.numpy(), y.numpy())
    # plt.title(label)
    # plt.grid()
# plt.tight_layout()
# plt.show() 

# GELU is smooth and differentiable everywhere, while ReLU has a sharp corner at x=0
# the small negative values produced by GELU can help maintain a richer gradient flow during backpropagation (contribute to learning process)

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), # 788 -> 3072 (4 times)
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]), # 3072 -> 768
        )

    def forward(self, x):
        return self.layers(x)

ffn = FeedForward(GPT_CONFIG_124)
x = torch.randn(2, 3, 768)
out = ffn(x)
# print(out.shape)

class ExampleDeepNeuralNetwork(nn.Module): # shortcut creates alternate path for gradient flow during backpropagation
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])
    
    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut:
                x = x + layer_output
            else:
                x = layer_output
        return x

layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([1., 0., -1.])
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)


def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.]])
    
    loss = nn.MSELoss()
    loss = loss(output, target)
    loss.backward()
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"Gradient for {name}: {param.grad.abs().mean().item()}")
        
# print_gradients(model_without_shortcut, sample_input)
# print_gradients(ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True), sample_input)
# Gradients with shortcut are generally larger, indicating better gradient flow and reduced vanishing gradient problem

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["dropout"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["dropout"])
    
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

torch.manual_seed(123)
x = torch.randn(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124)
output = block(x)
# print(x.shape)
# print(output.shape)

# dimension kept same so that we can stack multiple transformer blocks together
# output contains refined representations after attention and feed-forward processing
# attention helps in capturing contextual relationships between tokens in the sequence
# ffn helps in further transforming these representations for better learning, it relates the vector to meanings which are not even present in the sentence but are implied by context

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"]) # token embedding
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"]) # position embedding 
        self.drop_emb = nn.Dropout(cfg["dropout"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg)
              for _ in range(cfg["n_layers"])] # transformer blocks (n_layers times)
        )
        self.final_norm = LayerNorm(cfg["emb_dim"]) # final layer of normalization
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False   # output head to vocab size, where each dimension represents a token likelihood
        )
        
    def forward(self, x):
        batch_size, seq_length = x.shape
        tok_embeds = self.tok_emb(x)
        pos_embeds = self.pos_emb(
            torch.arange(seq_length, device=x.device)   
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124)
out = model(batch)
# print(batch)
# print(out)
# print(out.shape) # 2, 4, 50257

total_params = sum(p.numel() for p in model.parameters())
# print(f"Total parameters in GPT-124M model: {total_params}")

total_size_bytes = total_params * 4
total_mb = total_size_bytes / (1024 ** 2)
# print(f"Total size of GPT-124M model parameters: {total_mb:.2f} MB")

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] # last context_size tokens used as context
        logits = model(idx_cond)
        logits = logits[:, -1, :] # get logits for the last token, logits shape: (batch_size, vocab_size)
        probas = torch.softmax(logits, dim=-1) # convert logits to probabilities
        idx_next = torch.argmax(probas, dim=-1, keepdim=True) # get the token with highest probability
        idx = torch.cat((idx, idx_next), dim=1) # append to the context

    return idx

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
# print("encoded:", encoded)
encoded_tensor = torch.tensor([encoded]) # batch size of 1, shape: (1, seq_length)
# print("encoded tensor shape:", encoded_tensor.shape)
model.eval() # disables random components like dropout for deterministic output (which are only used in training)
out = generate_text_simple(model = model, idx = encoded_tensor, max_new_tokens = 6, context_size = GPT_CONFIG_124["context_length"])
# print("Output: ", out)
# print("Output length:", out.shape)

decoded_text = tokenizer.decode(out[0].tolist())
# print("Decoded text:", decoded_text)
