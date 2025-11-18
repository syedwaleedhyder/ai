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


class MultiHeadAttentionBlock(nn.Module):
    """
    Multi-Head Attention mechanism.
    Splits attention across multiple heads to capture different relationships.
    Query:  "The cat sat on the mat"
    Key:    "The cat sat on the mat"
    Value:  "The cat sat on the mat"

    Attention weights show which words attend to which:
        The  cat  sat  on   the  mat
    The   [0.9, 0.0, 0.0, 0.0, 0.1, 0.0]
    cat   [0.2, 0.6, 0.1, 0.0, 0.0, 0.1]
    sat   [0.0, 0.3, 0.5, 0.2, 0.0, 0.0]
    ...
    """
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h  # Number of heads
        
        # Ensure d_model is divisible by number of heads
        assert d_model % h == 0, 'd_model must be divisible by h'
        
        self.d_k = d_model // h  # Dimension per head
        
        # Weight matrices for Q, K, V, and output
        self.w_q = nn.Linear(d_model, d_model)  # Query weights
        self.w_k = nn.Linear(d_model, d_model)  # Key weights
        self.w_v = nn.Linear(d_model, d_model)  # Value weights
        self.w_o = nn.Linear(d_model, d_model)  # Output weights
        
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        Scaled Dot-Product Attention.
        Formula: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
        """
        d_k = query.shape[-1]
        # Calculate attention scores: QK^T / sqrt(d_k)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        # Apply mask (if provided) to prevent attention to certain positions
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        # Apply softmax to get attention weights
        attention_scores = attention_scores.softmax(dim=-1)
        # Apply dropout
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # Multiply by values and return
        return (attention_scores @ value), attention_scores
    
    def forward(self, q, k, v, mask):
        # Linear transformations
        query = self.w_q(q)  # (batch, seq_len, d_model)
        key = self.w_k(k)
        value = self.w_v(v)
        
        # Split into multiple heads
        # (batch, seq_len, d_model) → (batch, seq_len, h, d_k) 
        #                           → (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        
        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )
        
        # Concatenate heads
        # (batch, h, seq_len, d_k) → (batch, seq_len, h, d_k) 
        #                          → (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        
        # Final linear transformation
        return self.w_o(x)


class ResidualConnection(nn.Module):
    """
    Implements residual connection: LayerNorm(x + Sublayer(x))
    Helps with gradient flow in deep networks.
    """
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    
    def forward(self, x, sublayer):
        # Normalize first, then add to original input
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    """
    Single encoder block consisting of:
    1. Multi-head self-attention
    2. Feed-forward network
    Both with residual connections.
    """
    def __init__(
        self, 
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([
            ResidualConnection(dropout) for _ in range(2)
        ])
    
    def forward(self, x, src_mask):
        # Self-attention with residual connection
        x = self.residual_connections[0](
            x, 
            lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        """
        Inside the residual connecttion, the lambda expands to: self.self_attention_block(self.norm(x), self.norm(x), self.norm(x), src_mask)
        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))
        """

        # Feed-forward with residual connection
        x = self.residual_connections[1](x, self.feed_forward_block)
        
        return x


class Encoder(nn.Module):
    """
    Complete encoder: stack of N encoder blocks.
    """
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, mask):
        # Pass through all encoder blocks
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final normalization
        return self.norm(x)


class DecoderBlock(nn.Module):
    """
    Single decoder block consisting of:
    1. Masked multi-head self-attention
    2. Multi-head cross-attention (attends to encoder output)
    3. Feed-forward network
    All with residual connections.
    """
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([
            ResidualConnection(dropout) for _ in range(3)
        ])
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Masked self-attention
        x = self.residual_connections[0](
            x,
            lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )
        
        # Cross-attention (queries from decoder, keys/values from encoder)
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)
        )
        
        # Feed-forward
        x = self.residual_connections[2](x, self.feed_forward_block)
        
        return x
    

class Decoder(nn.Module):
    """
    Complete decoder: stack of N decoder blocks.
    """
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Pass through all decoder blocks
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        # Final normalization
        return self.norm(x)
    

class ProjectionLayer(nn.Module):
    """
    Projects decoder output to vocabulary size and applies log softmax.
    """
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # (batch, seq_len, d_model) → (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)

class Transformer(nn.Module):
    """
    Complete Transformer model with encoder and decoder.
    """
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        source_embeddings: InputEmbeddings,
        target_embeddings: InputEmbeddings,
        positional_encoding: PositionalEncoding,
        projection_layer: ProjectionLayer
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embeddings = source_embeddings
        self.target_embeddings = target_embeddings
        self.positional_encoding = positional_encoding
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # Embed and encode source sequence
        src_embedded = self.positional_encoding(self.source_embeddings(src))
        return self.encoder(src_embedded, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        # Embed and decode target sequence
        tgt_embedded = self.positional_encoding(self.target_embeddings(tgt))
        return self.decoder(tgt_embedded, encoder_output, src_mask, tgt_mask)
    
    def project(self, decoder_output):
        # Project to vocabulary size
        return self.projection_layer(decoder_output)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        # Full forward pass through encoder, decoder, and projection
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(encoder_output, src_mask, tgt, tgt_mask)
        return self.project(decoder_output)
    

def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048
) -> Transformer:
    """
    Helper function to build a complete Transformer model.
    """
    # Create source and target embedding layers
    source_embeddings = InputEmbeddings(d_model, src_vocab_size)
    target_embeddings = InputEmbeddings(d_model, tgt_vocab_size)

    # Create positional encoding layers
    src_positional_encoding = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_positional_encoding = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create encoder blocks
    encoder_layers = nn.ModuleList([
        EncoderBlock(
            self_attention_block=MultiHeadAttentionBlock(d_model, h, dropout),
            feed_forward_block=FeedForwardBlock(d_model, d_ff, dropout),
            dropout=dropout
        ) for _ in range(N)
    ])
    encoder = Encoder(encoder_layers)
    
    # Create decoder blocks
    decoder_layers = nn.ModuleList([
        DecoderBlock(
            self_attention_block=MultiHeadAttentionBlock(d_model, h, dropout),
            cross_attention_block=MultiHeadAttentionBlock(d_model, h, dropout),
            feed_forward_block=FeedForwardBlock(d_model, d_ff, dropout),
            dropout=dropout
        ) for _ in range(N)
    ])
    decoder = Decoder(decoder_layers)
    
    # Create projection layer
    projection_layer = ProjectionLayer(d_model, vocab_size)
    
    # Build the complete Transformer model
    transformer = Transformer(
        encoder=encoder,
        decoder=decoder,
        source_embeddings=source_embeddings,
        target_embeddings=target_embeddings,
        src_positional_encoding=src_positional_encoding,
        tgt_positional_encoding=tgt_positional_encoding,
        projection_layer=projection_layer
    )

    # Initialize weights (optional, can use default initialization)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


    return transformer

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

    # MultiHeadAttention test
    mha = MultiHeadAttentionBlock(d_model=d_model, h=4, dropout=0.1)
    out_mha = mha(out_ln, out_ln, out_ln, mask=None)
    print("After MultiHeadAttention shape:", out_mha.shape)
    print("First MHA vector (first 5 dims):", out_mha[0, 0, :5].tolist())
    if hasattr(mha, 'attention_scores') and mha.attention_scores is not None:
        print("Attention scores shape:", mha.attention_scores.shape)
        print("Attention scores sample (batch 0, head 0, query 0, first 5 keys):", mha.attention_scores[0, 0, 0, :5].tolist())
    else:
        print("No attention scores available")

    # ResidualConnection tests
    residual = ResidualConnection(dropout=0.1)
    out_res_ff = residual(out_ln, lambda y: ff(y))
    print("After ResidualConnection around FF shape:", out_res_ff.shape)
    print("Residual-FF first 5 dims:", out_res_ff[0, 0, :5].tolist())

    out_res_mha = residual(out_ln, lambda y: mha(y, y, y, mask=None))
    print("After ResidualConnection around MHA shape:", out_res_mha.shape)
    print("Residual-MHA first 5 dims:", out_res_mha[0, 0, :5].tolist())
    if hasattr(mha, 'attention_scores') and mha.attention_scores is not None:
        print("Residual MHA attention shape:", mha.attention_scores.shape)

    # EncoderBlock test
    enc_block = EncoderBlock(
        self_attention_block=MultiHeadAttentionBlock(d_model=d_model, h=4, dropout=0.1),
        feed_forward_block=FeedForwardBlock(d_model=d_model, d_ff=64, dropout=0.1),
        dropout=0.1,
    )
    out_enc_block = enc_block(out_ln, src_mask=None)
    print("After EncoderBlock shape:", out_enc_block.shape)
    print("EncoderBlock first 5 dims:", out_enc_block[0, 0, :5].tolist())

    # Encoder (stack of 2 encoder blocks) test
    layers = nn.ModuleList([
        EncoderBlock(
            self_attention_block=MultiHeadAttentionBlock(d_model=d_model, h=4, dropout=0.1),
            feed_forward_block=FeedForwardBlock(d_model=d_model, d_ff=64, dropout=0.1),
            dropout=0.1,
        ) for _ in range(2)
    ])
    encoder = Encoder(layers)
    out_encoder = encoder(out_ln, mask=None)
    print("After Encoder (2 layers) shape:", out_encoder.shape)
    print("Encoder first 5 dims:", out_encoder[0, 0, :5].tolist())

    # DecoderBlock test
    # Use embeddings as decoder inputs for the test
    tgt = out.clone()
    # Causal mask to prevent attending to future positions
    tgt_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

    dec_block = DecoderBlock(
        self_attention_block=MultiHeadAttentionBlock(d_model=d_model, h=4, dropout=0.1),
        cross_attention_block=MultiHeadAttentionBlock(d_model=d_model, h=4, dropout=0.1),
        feed_forward_block=FeedForwardBlock(d_model=d_model, d_ff=64, dropout=0.1),
        dropout=0.1,
    )
    out_dec_block = dec_block(tgt, encoder_output=out_encoder, src_mask=None, tgt_mask=tgt_mask)
    print("After DecoderBlock shape:", out_dec_block.shape)
    print("DecoderBlock first 5 dims:", out_dec_block[0, 0, :5].tolist())
    if hasattr(dec_block.cross_attention_block, 'attention_scores') and dec_block.cross_attention_block.attention_scores is not None:
        print("Decoder cross-attention scores shape:", dec_block.cross_attention_block.attention_scores.shape)
        print("Decoder cross-attention sample (batch 0, head 0, query 0, first 5 keys):", dec_block.cross_attention_block.attention_scores[0, 0, 0, :5].tolist())

    # Decoder (stack of 2 decoder blocks) test
    dec_layers = nn.ModuleList([
        DecoderBlock(
            self_attention_block=MultiHeadAttentionBlock(d_model=d_model, h=4, dropout=0.1),
            cross_attention_block=MultiHeadAttentionBlock(d_model=d_model, h=4, dropout=0.1),
            feed_forward_block=FeedForwardBlock(d_model=d_model, d_ff=64, dropout=0.1),
            dropout=0.1,
        ) for _ in range(2)
    ])
    decoder = Decoder(dec_layers)
    out_decoder = decoder(tgt, encoder_output=out_encoder, src_mask=None, tgt_mask=tgt_mask)
    print("After Decoder (2 layers) shape:", out_decoder.shape)
    print("Decoder first 5 dims:", out_decoder[0, 0, :5].tolist())

    # ProjectionLayer test
    projection = ProjectionLayer(d_model=d_model, vocab_size=vocab_size)
    out_proj = projection(out_decoder)  # log-probabilities over vocab
    print("After ProjectionLayer (log probs) shape:", out_proj.shape)
    # Check that probabilities sum to ~1 across vocab for a few positions
    prob_sums = torch.exp(out_proj).sum(dim=-1)  # (batch, seq_len)
    print("Sum of probabilities (should be 1) for first batch:", prob_sums[0].tolist())
    # Show top-3 tokens for first batch, first token
    topk_vals, topk_idx = torch.topk(out_proj[0, 0], k=3)
    print("Top-3 token indices (first batch, first token):", topk_idx.tolist())
    print("Top-3 log-probs:", topk_vals.tolist())

    # Complete Transformer test
    transformer = build_transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        src_seq_len=seq_len,
        tgt_seq_len=seq_len,
    )
    out_transformer = transformer.encode(x, src_mask=None)
    print("After Transformer encode shape:", out_transformer.shape)
    out_transformer_dec = transformer.decode(
        encoder_output=out_transformer,
        src_mask=None,
        tgt=x,
        tgt_mask=tgt_mask
    )
    print("After Transformer decode shape:", out_transformer_dec.shape)
    out_transformer_proj = transformer.project(out_transformer_dec)
    print("After Transformer project (log probs) shape:", out_transformer_proj.shape)
    prob_sums_trans = torch.exp(out_transformer_proj).sum(dim=-1)  # (batch, seq_len)
    print("Sum of probabilities (should be 1) for first batch (Transformer):", prob_sums_trans[0].tolist())
    topk_vals_trans, topk_idx_trans = torch.topk(out_transformer_proj[0, 0], k=3)
    print("Top-3 token indices (first batch, first token) Transformer:", topk_idx_trans.tolist())
    print("Top-3 log-probs (Transformer):", topk_vals_trans.tolist())
    # Full forward pass
    out_transformer_full = transformer(x, x, src_mask=None, tgt_mask=tgt_mask)
    print("After Transformer full forward (log probs) shape:", out_transformer_full.shape)
    prob_sums_full = torch.exp(out_transformer_full).sum(dim=-1)  # (batch, seq_len)
    print("Sum of probabilities (should be 1) for first batch (Transformer full):", prob_sums_full[0].tolist())
    topk_vals_full, topk_idx_full = torch.topk(out_transformer_full[0, 0], k=3)
    print("Top-3 token indices (first batch, first token) Transformer full:", topk_idx_full.tolist())
    print("Top-3 log-probs (Transformer full):", topk_vals_full.tolist())

