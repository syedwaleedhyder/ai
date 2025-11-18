# Flash Attention From First Principles

This repo documents my step-by-step journey of learning and implementing **Flash Attention** from basics to full CUDA/Triton kernels.

## Topics

- Introduction to Multi-Head Attention
- Safe Softmax
- Online Softmax
- Introduction to CUDA \& GPUs
- Tensor Layouts
- Example of CUDA Kernels
- Block Matrix Multiplication
- From CUDA to Triton
- Software Pipelining
- Flash Attention (Forward Pass)
- Autograd
- Derivatives and Gradients
- Gradient of the MatMul operation
- Gradient of the softmax operation
- Flash Attention (Backward Pass)

### Introduction to Multi-Head Attention
Source: [Coding a Transformer from scratch on PyTorch, with full explanation, training and inference.](https://www.youtube.com/watch?v=ISNdQcPhsts)

#### Introduction
<img src="resources/transformer_architecture.png" alt="transformer_architecture" height="400" />

#### Input Embeddings
<img src="resources/input_embedding.png" alt="input embeddings" height="400" />

#### Positional Encodings
<img src="resources/positional_encoding.png" alt="positional_encoding" height="400" />
<img src="resources/positional_encoding_2.png" alt="positional_encoding_2" height="400" />

#### Log Transform in Positional Encoding

- **Math identity:** $ a^b = e^{b \ln(a)} $
- $ 10000^{\frac{2i}{d_{model}}} = e^{\frac{2i}{d_{model}} \ln(10000)} $
- Invert for denominator: $ 10000^{-\frac{2i}{d_{model}}} = e^{-\frac{2i}{d_{model}} \ln(10000)} $
- **Code:**

```python
div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
```

- This matches the scaling in the original formula.

#### Layer Normalization
<img src="resources/layer_normalization.png" alt="layer_normalization" height="400" />

#### Feed Forward
<img src="resources/feed_forward.png" alt="feed_forward" height="400" />

#### Multi-Head Attention
<img src="resources/multihead_attention.png" alt="multihead_attention" height="400" />

#### Residual Connection
<img src="resources/residual_connection.png" alt="residual_connection" height="400" />

#### Encoder
<img src="resources/encoder.png" alt="encoder" height="400" />

#### Decoder
<img src="resources/decoder.png" alt="decoder" height="400" />

#### Linear Layer
Converts decoder output to vocabulary probabilities.
#### Transformer
<!-- ![transformer_architecture](resources/transformer_architecture.png) -->
<img src="resources/transformer_architecture.png" alt="transformer_architecture" height="400" />

#### Task overview
#### Tokenizer
#### Dataset
#### Training loop
#### Validation loop
#### Attention visualization



**Inspired by:**
[Flash Attention from First Principles (YouTube)](https://www.youtube.com/watch?v=zy8ChVd_oTM)
