import torch
import torch.nn as nn

if __name__ == "__main__":
    inputs = torch.tensor(
      [[0.43, 0.15, 0.89],
       [0.55, 0.87, 0.66],
       [0.57, 0.85, 0.64],
       [0.22, 0.58, 0.33],
       [0.77, 0.25, 0.10],
       [0.05, 0.80, 0.55]
       ])

    query = inputs[1]

    attn_scores_2 = torch.empty(inputs.shape[0]) # tensor of inputs size containing uninitialized values

    for i,x_i in enumerate(inputs):
        attn_scores_2[i] = torch.dot(query, x_i) # query . key for each key, gives similarity score
        
    # print(attn_scores_2)

    # find attention weights by applying normalization (softmax)

    # attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
    # print(attn_weights_2_tmp)

    # def softmax(x): # can give overflow / underflow issues
        # return torch.exp(x) / torch.exp(x).sum(dim = 0)

    # attn_weights_2 = softmax(attn_scores_2)

    attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
    # print(attn_weights_2)

    context_vector_2 = torch.zeros(query.shape)

    for i, x_i in enumerate(inputs):
        context_vector_2 += attn_weights_2[i] * x_i
        
    # print(context_vector_2)

    attn_scores = torch.empty(6, 6)

    # for i, x_i in enumerate(inputs):
    #     for j, x_j in enumerate(inputs):
    #         attn_scores[i, j] = torch.dot(x_i, x_j)

    attn_scores = inputs @ inputs.T
    # print(attn_scores)

    attn_weights = torch.softmax(attn_scores, dim=1) # dim = 1 since we want each row to sum to 1, so we normalize across columns
    # print(attn_weights)

    # print(attn_weights.sum(dim=1))

    all_context_vectors = attn_weights @ inputs
    # print(all_context_vectors)

    # self-attention with trainable weights

    x_2 = inputs[1]
    d_in = inputs.shape[1]
    d_out = 2

    torch.manual_seed(123)
    W_q = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_k = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_v = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    # created random weight matrices for query, key, value

    query_2 = x_2 @ W_q
    keys = inputs @ W_k
    values = inputs @ W_v

    # ey_2 = keys[1]
    # attn_scores_2 = query_2.dot(key_2)
    # print(attn_scores_2)

    attn_scores_2 = query_2 @ keys.T
    # print(attn_scores_2)

    d_k = keys.shape[-1]
    attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim = -1) # force the variance to be 1 (scaling factor)
    # print(attn_weights_2)

    context_vector_2 = attn_weights_2 @ values
    # print(context_vector_2)


class SelfAttention_v1((nn.Module)):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_q = nn.Parameter(torch.rand(d_in, d_out))
        self.W_k = nn.Parameter(torch.rand(d_in, d_out))
        self.W_v = nn.Parameter(torch.rand(d_in, d_out))
        
    def forward(self, inputs):
        queries = inputs @ self.W_q
        keys = inputs @ self.W_k
        values = inputs @ self.W_v
        
        d_k = keys.shape[-1]
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)
        context_vectors = attn_weights @ values
        return context_vectors
    
if __name__ == "__main__":
    torch.manual_seed(123)
    sa_v1 = SelfAttention_v1(d_in=3, d_out=2)
    # print(sa_v1(inputs))

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias) # better as they handle weight initialization internally
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, inputs):
        queries = self.W_q(inputs)
        keys = self.W_k(inputs)
        values = self.W_v(inputs)
        
        d_k = keys.shape[-1]
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)
        context_vectors = attn_weights @ values
        return context_vectors
    
if __name__ == "__main__":
    torch.manual_seed(789)
    sa_v2 = SelfAttention_v2(d_in=3, d_out=2)
    # print(sa_v2(inputs))


    # casual self-attention (masking future tokens)
    queries = sa_v2.W_q(inputs)
    keys = sa_v2.W_k(inputs)
    attn_scores = queries @ keys.T
    attn_weights = torch.softmax(attn_scores / (keys.shape[-1]**0.5), dim=-1)
    # print(attn_weights)

    context_length = attn_scores.shape[0]
    mask_simple = torch.tril(torch.ones((context_length, context_length)))
    # print(mask_simple)

    masked_simple = attn_weights * mask_simple
    # print(masked_simple)

    # renormalize after masking for sum per row to be 1

    row_sums = masked_simple.sum(dim = -1, keepdim=True)
    masked_simple_norm = masked_simple / row_sums
    # print(masked_simple_norm)

    mask = torch.triu(torch.ones((context_length, context_length)) * float('-inf'), diagonal = 1) # e^-inf = 0
    masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
    # print(masked)

    attn_weights = torch.softmax(masked / (keys.shape[-1]**0.5), dim = -1)
    # print(attn_weights)


    # reducing overfitting (dropout)
    torch.manual_seed(123)
    dropout = torch.nn.Dropout(0.5)
    example = torch.ones(6, 6)
    # print(dropout(example)) # matrix of 2s and 0s as pytorch scales up the remaining values to maintain expected sum (loudness) -> 1 / 0.5

    torch.manual_seed(123)
    # print(dropout(attn_weights))

    batch = torch.stack((inputs, inputs), dim=0) # like 2 sentenes
    # print(batch.shape)

class CasualAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones((context_length, context_length)) * float('-inf'), diagonal = 1)) # buffers are automatically moved to GPU with model

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.d_out**0.5, dim = -1)
        attn_weights = self.dropout(attn_weights)
        
        context_vec = attn_weights @ values
        return context_vec
    
if __name__ == "__main__":
    torch.manual_seed(123)
    context_length = batch.shape[1]
    ca = CasualAttention(d_in=3, d_out=2, context_length=context_length, dropout=0.0)
    context_vecs = ca(batch)
    # print(context_vecs)

# multi - head attention

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias = False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CasualAttention(d_in, d_out, context_length, dropout ,qkv_bias)
            for i in range(num_heads)]
        )
        
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim = -1)
    
if __name__ == "__main__":
    torch.manual_seed(123)
    context_length = batch.shape[1]
    d_in, d_out = 3, 2
    mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, dropout=0.0, num_heads=2)
    context_vecs = mha(batch)

    # print(context_vecs) # 2 dimn from each head concatenated
    # print(context_vecs.shape)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False): # context length is max sequence length for the mask
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # dimension per head
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones((context_length, context_length)) * float('-inf'), diagonal=1))
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) # reshape for multi-head
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        
        keys.transpose_(1, 2) # move head dimension to the front so that it is treated as batch dimension
        queries.transpose_(1, 2)
        values.transpose_(1, 2)
        
        attn_scores = queries @ keys.transpose(2, 3) # flip last two dimensions for dot product
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2).contiguous().view(b, num_tokens, self.d_out) # reshape back to original
        context_vec = self.out_proj(context_vec) # final linear layer to mix heads
        return context_vec
    
if __name__ == "__main__":
    torch.manual_seed(123)
    batch_size, context_length, d_in = batch.shape
    d_out = 2
    mha = MultiHeadAttention(d_in, d_out, context_length, dropout = 0.0, num_heads = 2)
    context_vecs = mha(batch)
    # print(context_vecs)
    # print(context_vecs.shape)