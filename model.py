import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------

class Block(nn.Module): # Assembler Class for attention. Made for repeatability

    def __init__(self, dim, n_query_heads, n_kv_heads, hidden_dim):
        super().__init__()
        self.attention_norm = RMSNorm(dim) # cleans numbers so size is consistent
        self.attention = GroupedQueryAttention(dim, n_query_heads, n_kv_heads) # tokens communicate
        self.ffn_norm = RMSNorm(dim) # cleans numbers so size is consistent
        self.feed_forward = SwiGLU(dim, hidden_dim) # tokens think privatley on information recieved

    def forward(self, x): # One full transformer layer. - Norm, Attention, add back, Norm, FeedForward, add back
        x = x + self.attention(self.attention_norm(x)) # clean x in norm, run attention on clean copy, add back to og x
        x = x + self.feed_forward(self.ffn_norm(x)) # clean new x in norm, run feedforward on new clean copy, add back to x
        return x # return twice adjested token

# -----------------------------------------------------

class Solu(nn.Module):
    def __init__(self, vocab_size, dim, n_layers, n_query_heads, n_kv_heads, hidden_dim, max_seq_len=1024):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim) # lookup table. vocab number -> 576 array of numbers to represent token
        self.layers = nn.ModuleList([ # holds list of layers
            Block(dim, n_query_heads, n_kv_heads, hidden_dim)
            for _ in range(n_layers) # creates n_layers(30) number of Block Objects
        ])
        self.final_norm = RMSNorm(dim) # final cleanup after 30 Blocks
        self.output = nn.Linear(dim, vocab_size, bias=False) # tokens 576-vector -> 49152 scores, one per possible next token
        self.output.weight = self.token_embedding.weight # reuse embedding table for output. saves 28M params
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():  # if linear or embedding, sets weight to normal and standard dev to 0.02
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        n_layers = len(self.layers)
        scale = 0.02 / (2 * n_layers) ** 0.5
        for block in self.layers: # sets standard dev for o_proj and down_layer to scale
            nn.init.normal_(block.attention.o_proj.weight, mean=0.0, std=scale)
            nn.init.normal_(block.feed_forward.down_layer.weight, mean=0.0, std=scale)

    def forward(self, idx):
        x = self.token_embedding(idx) # converts token ID's to vectors

        for layer in self.layers: # tokens run through 30 Blocks in order
            x = layer(x)

        x = self.final_norm(x)
        logits = self.output(x) # produces raw prediction scores | not yet probabilities |
        return logits

# -----------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps) # initialize RMSNorm function
        normalized = x / rms # apply RMSNorm function to normalize
        return normalized * self.weight

# -----------------------------------------------------

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_layer = nn.Linear(dim, hidden_dim, bias=False)
        self.content_layer = nn.Linear(dim, hidden_dim, bias=False)
        self.down_layer = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_layer(x)) # silu gate
        content = self.content_layer(x) # keep content
        fused = gate * content # multiply silu'd gate by content
        return self.down_layer(fused) # reduce back to og matrix layer

# -----------------------------------------------------

class GroupedQueryAttention(nn.Module):
    def __init__(self, dim, n_query_heads, n_kv_heads, max_seq_len=1024, theta=10000.0, use_flash=True):
        super().__init__()
        self.n_query_heads = n_query_heads # 9
        self.n_kv_heads = n_kv_heads # 3
        self.head_dim = dim // n_query_heads # 576 // 9 = 64
        self.n_groups = n_query_heads // n_kv_heads # 9 // 3 = 3
        self.use_flash = use_flash

        self.q_proj = nn.Linear(dim, n_query_heads * self.head_dim, bias=False) # (576, 9 * 64) -> (576, 576)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False) # (576, 3 * 64) -> (576, 192)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False) # (576, 3 * 64) -> (576, 192)
        self.o_proj = nn.Linear(n_query_heads * self.head_dim, dim, bias=False) # (9 * 64, 576) -> (576, 576)

        freqs = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)) # RoPE rotation speeds
        self.register_buffer("freqs", freqs) # store speeds as a fixed buffer

        positions = torch.arange(max_seq_len).float()
        angles = torch.outer(positions, freqs)
        self.register_buffer("cos", torch.cos(angles))
        self.register_buffer("sin", torch.sin(angles))

    def forward(self, x):
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_query_heads, self.head_dim) # (B, T, 9, 64)
        k = k.view(B, T, self.n_kv_heads, self.head_dim) # (B, T, 3, 64)
        v = v.view(B, T, self.n_kv_heads, self.head_dim) # (B, T, 3, 64)

        q = self.apply_rope(q, T)
        k = self.apply_rope(k, T)

        k = k.repeat_interleave(self.n_groups, dim=2) # repeat k 3 times. dim=2 because Head is in dim 2
        v = v.repeat_interleave(self.n_groups, dim=2) # repeat v 3 times. dim=2 because Head is in dim 2

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        if self.use_flash:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            scores = q @ k.transpose(-2, -1)
            scores = scores / (self.head_dim ** 0.5)

            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))

            weights = F.softmax(scores, dim=-1)

            out = weights @ v
        out = out.transpose(1,2).contiguous().view(B, T, C) # swaps Head and Token dimensions back. (Merges each token's 9 heads of 64 numbers into 576 number-vector)
        out = self.o_proj(out)
        return out # (B, T, 576)

    def apply_rope(self, x, T):
        cos = self.cos[:T][None, :, None, :] # grabs first T positions from precomputed table
        sin = self.sin[:T][None, :, None, :] # grabs first T positions from precomputed table

        x1 = x[..., :self.head_dim // 2].float()
        x2 = x[..., self.head_dim // 2:].float()

        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        return torch.cat([rotated_x1, rotated_x2], dim=-1).to(x.dtype)

# -----------------------------------------------------

# CHECKS/TESTS
if __name__ == "__main__":
    model = Solu( # Builds model shape
        vocab_size=49152,
        dim=576,
        n_layers=30,
        n_query_heads=9,
        n_kv_heads=3,
        hidden_dim=1536,
    )

    total_params = sum(p.numel() for p in model.parameters()) # counts parameters
    print(f"Total parameters: {total_params:,}") # 134,515,008
    print(f"In millions: {total_params / 1e6:.1f}M") # 134.5M

    idx = torch.randint(0, 49152, (2, 100)) # creates fake batch of token ID's -> 2 seqences of 100 random token ID's
    logits = model(idx) # run full foward for confimration
    print("input shape: ", idx.shape) # input shape:  torch.Size([2, 100])
    print("output shape:", logits.shape) # output shape: torch.Size([2, 100, 49152])
    # verify flash and hand-built attention agree
    torch.manual_seed(0)
    test_attn_flash = GroupedQueryAttention(576, 9, 3, use_flash=True)
    test_attn_manual = GroupedQueryAttention(576, 9, 3, use_flash=False)
    test_attn_manual.load_state_dict(test_attn_flash.state_dict())

    x_test = torch.randn(2, 100, 576)
    out_flash = test_attn_flash(x_test)
    out_manual = test_attn_manual(x_test)
    max_diff = (out_flash - out_manual).abs().max().item()
    print(f"max difference between flash and manual attention: {max_diff}")
