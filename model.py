import math
import torch

class InputEmbeddings(torch.nn.Module):
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
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        # Normalizing the variance of the embeddings
        return self.embedding(x) * math.sqrt(self.d_model)

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

    # Forward pass
    out = model(x)

    # Print diagnostics
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
