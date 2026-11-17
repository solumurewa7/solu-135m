"""
Solu-135M model definition.

A 135M-parameter decoder-only transformer (GPT-2 scale, modern architecture:
grouped-query attention, RoPE, RMSNorm, SwiGLU, tied embeddings). Implemented
from scratch in PyTorch; only tensor ops and autograd are used, no high-level
transformer modules.

This file defines the architecture and is shared by training and inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (no mean subtraction, no bias)."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network: SiLU(gate) * content, then project down."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_layer = nn.Linear(dim, hidden_dim, bias=False)
        self.content_layer = nn.Linear(dim, hidden_dim, bias=False)
        self.down_layer = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_layer(x))
        content = self.content_layer(x)
        return self.down_layer(gate * content)


class GroupedQueryAttention(nn.Module):
    """
    Grouped-query attention with rotary position embeddings.

    n_query_heads query heads share n_kv_heads key/value heads
    (here 9 query heads over 3 KV heads, 3 query heads per KV head). This cuts
    the size of the K/V projections while keeping full query resolution.

    Args:
        dim:           model dimension.
        n_query_heads: number of query heads.
        n_kv_heads:    number of key/value heads (must divide n_query_heads).
        max_seq_len:   longest sequence the RoPE tables are precomputed for.
        theta:         RoPE base frequency.
        use_flash:     if True, use PyTorch's fused scaled_dot_product_attention;
                       if False, use the explicit attention implementation below
                       (kept for reference / verification, identical output).
    """

    def __init__(self, dim, n_query_heads, n_kv_heads, max_seq_len=1024, theta=10000.0, use_flash=True):
        super().__init__()
        self.n_query_heads = n_query_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_query_heads
        self.n_groups = n_query_heads // n_kv_heads   # query heads per KV head
        self.use_flash = use_flash

        self.q_proj = nn.Linear(dim, n_query_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_query_heads * self.head_dim, dim, bias=False)

        # Precompute RoPE cos/sin tables up to max_seq_len.
        freqs = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("freqs", freqs)
        positions = torch.arange(max_seq_len).float()
        angles = torch.outer(positions, freqs)
        self.register_buffer("cos", torch.cos(angles))
        self.register_buffer("sin", torch.sin(angles))

    def forward(self, x):
        # x: (batch, seq_len, dim)
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_query_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = self.apply_rope(q, T)
        k = self.apply_rope(k, T)

        # Expand each KV head to match its group of query heads.
        k = k.repeat_interleave(self.n_groups, dim=2)
        v = v.repeat_interleave(self.n_groups, dim=2)

        # (B, T, heads, head_dim) -> (B, heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.use_flash:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))
            weights = F.softmax(scores, dim=-1)
            out = weights @ v

        # Merge heads back to (B, T, dim) and project out.
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)

    def apply_rope(self, x, T):
        """Apply rotary position embedding to x (B, T, heads, head_dim).

        Rotation is computed in float32 even under mixed precision, since
        bfloat16 loses positional accuracy at longer context lengths.
        """
        cos = self.cos[:T][None, :, None, :]
        sin = self.sin[:T][None, :, None, :]

        x1 = x[..., :self.head_dim // 2].float()
        x2 = x[..., self.head_dim // 2:].float()

        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        return torch.cat([rotated_x1, rotated_x2], dim=-1).to(x.dtype)


class Block(nn.Module):
    """One pre-norm transformer layer: attention sublayer then feed-forward sublayer,
    each wrapped in a residual connection."""

    def __init__(self, dim, n_query_heads, n_kv_heads, hidden_dim):
        super().__init__()
        self.attention_norm = RMSNorm(dim)
        self.attention = GroupedQueryAttention(dim, n_query_heads, n_kv_heads)
        self.ffn_norm = RMSNorm(dim)
        self.feed_forward = SwiGLU(dim, hidden_dim)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class Solu(nn.Module):
    """
    The full Solu-135M model: token embedding -> n_layers transformer blocks ->
    final norm -> output projection over the vocabulary.

    The output projection shares weights with the token embedding (tied
    embeddings), which removes a separate ~28M-parameter output matrix.
    """

    def __init__(self, vocab_size, dim, n_layers, n_query_heads, n_kv_heads, hidden_dim, max_seq_len=1024):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            Block(dim, n_query_heads, n_kv_heads, hidden_dim)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight   # tied embeddings
        self._init_weights()

    def _init_weights(self):
        # GPT-2 style init: std 0.02 everywhere, with residual-writing
        # projections scaled down by 1/sqrt(2 * n_layers) so activation
        # variance stays stable across the residual stream.
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        n_layers = len(self.layers)
        scale = 0.02 / (2 * n_layers) ** 0.5
        for block in self.layers:
            nn.init.normal_(block.attention.o_proj.weight, mean=0.0, std=scale)
            nn.init.normal_(block.feed_forward.down_layer.weight, mean=0.0, std=scale)

    def forward(self, idx):
        # idx: (batch, seq_len) token ids -> logits (batch, seq_len, vocab_size)
        x = self.token_embedding(idx)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return self.output(x)


# Smoke test: build the model, check the parameter count and output shape,
# and confirm the fused and explicit attention paths agree.
if __name__ == "__main__":
    model = Solu(
        vocab_size=49152,
        dim=576,
        n_layers=30,
        n_query_heads=9,
        n_kv_heads=3,
        hidden_dim=1536,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")       # expect 134,515,008
    print(f"In millions: {total_params / 1e6:.1f}M")

    idx = torch.randint(0, 49152, (2, 100))
    logits = model(idx)
    print("input shape: ", idx.shape)                  # (2, 100)
    print("output shape:", logits.shape)               # (2, 100, 49152)

    # Fused vs explicit attention should match to floating-point tolerance.
    torch.manual_seed(0)
    attn_flash = GroupedQueryAttention(576, 9, 3, use_flash=True)
    attn_manual = GroupedQueryAttention(576, 9, 3, use_flash=False)
    attn_manual.load_state_dict(attn_flash.state_dict())

    x_test = torch.randn(2, 100, 576)
    max_diff = (attn_flash(x_test) - attn_manual(x_test)).abs().max().item()
    print(f"max difference between flash and manual attention: {max_diff}")
