import math
import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):
    """
    Converts input tokens to embeddings of dimension d_model.
    Embeddings are normalized by multiplying by sqrt(d_model).
    Token IDs: [45, 123, 789, 12]
            ↓ (Embedding Layer)
    Vectors: [[0.23, -0.45, ...], [0.67, 0.12, ...], ...]
            (Each vector has d_model=512 dimensions)
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model  # Dimension of vectors (512)
        self.vocab_size = vocab_size  # Size of the vocabulary
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        # Normalizing the variance of the embeddings
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """
    Adds positional information to embeddings using sine/cosine functions.
    This allows the model to understand token positions in the sequence.
    """
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create positional encoding matrix (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create position indices [0, 1, 2, ..., seq_len-1]
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # Calculate division term for the formula
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add batch dimension
        pe = pe.unsqueeze(0)
        # Register as buffer (not a trainable parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # Add positional encoding to input
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    """
    Normalizes inputs across features for stable training.
    Uses learnable parameters alpha (scale) and bias (shift).
    """
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps  # Small value to avoid division by zero
        self.alpha = nn.Parameter(torch.ones(1))  # Learnable scale
        self.bias = nn.Parameter(torch.zeros(1))  # Learnable shift
    
    def forward(self, x):
        # Calculate mean and std across the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        
        # Normalize: (x - mean) / std, then scale and shift
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    """
    Position-wise feed-forward network.
    Formula: FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # Expand dimension
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # Project back
    
    def forward(self, x):
        # Shape: (batch, seq_len, d_model) → (batch, seq_len, d_ff) 
        #        → (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


if __name__ == "__main__":
    # Reproducible results for the test run
    torch.manual_seed(0)

    # Small test configuration
    batch_size = 2
    seq_len = 4
    vocab_size = 1000
    d_model = 16

    # Initialize model and sample input token ids
    model = InputEmbeddings(d_model=d_model, vocab_size=vocab_size)
    x = torch.tensor([[45, 123, 789, 12], [1, 2, 3, 4]], dtype=torch.long)

    # Forward pass through token embeddings
    out = model(x)

    # Print diagnostics for embeddings
    print("Input tokens shape:", x.shape)
    print("Embedding weight shape:", model.embedding.weight.shape)
    print("Output embeddings shape:", out.shape)
    print("d_model:", model.d_model)
    print("Output dtype:", out.dtype)
    print("Output device:", out.device)
    print("First output vector (first batch, first token) sample:", out[0, 0, :5].tolist())
    print("Embedding (raw) sample for token 45 (first 5 dims):", model.embedding.weight[45, :5].tolist())
    print("Embedding (scaled) sample for token 45 (first 5 dims):", (model.embedding.weight[45, :5] * math.sqrt(model.d_model)).tolist())
    print("Mean of outputs:", out.mean().item())
    print("Std of outputs:", out.std().item())

    # Positional Encoding test
    pos_enc = PositionalEncoding(d_model=d_model, seq_len=seq_len, dropout=0.0)
    out_pe = pos_enc(out)
    print("After PositionalEncoding shape:", out_pe.shape)
    print("First pos-encoded vector (first 5 dims):", out_pe[0, 0, :5].tolist())

    # Layer Normalization test
    ln = LayerNormalization(eps=1e-6)
    out_ln = ln(out_pe)
    print("After LayerNormalization shape:", out_ln.shape)
    # Print mean/std across features for first batch, first token (should be near 0/1)
    first_mean = out_ln.mean(dim=-1)[0, 0].item()
    first_std = out_ln.std(dim=-1)[0, 0].item()
    print("LayerNorm mean (first batch, first token):", first_mean)
    print("LayerNorm std  (first batch, first token):", first_std)
    print("LayerNorm alpha:", ln.alpha.item(), "bias:", ln.bias.item())

    # FeedForwardBlock test
    ff = FeedForwardBlock(d_model=d_model, d_ff=64, dropout=0.1)
    out_ff = ff(out_ln)
    print("After FeedForwardBlock shape:", out_ff.shape)
    print("First FF vector (first 5 dims):", out_ff[0, 0, :5].tolist())
    print("FeedForward mean:", out_ff.mean().item())
    print("FeedForward std:", out_ff.std().item())
