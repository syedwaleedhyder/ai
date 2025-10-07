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
![transformer_architecture](resources/transformer_architecture.png)
#### Input Embeddings
![input embeddings](resources/input_embedding.png)
#### Positional Encodings
![positional_encoding](resources/positional_encoding.png)
![positional_encoding_2](resources/positional_encoding_2.png)

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
![layer_normalization](resources/layer_normalization.png)
#### Feed Forward
![feed_forward](resources/feed_forward.png)
#### Multi-Head Attention
![multihead_attention](resources/multihead_attention.png)
#### Residual Connection
![residual_connection](resources/residual_connection.png)
#### Encoder
![encoder](resources/encoder.png)
#### Decoder
![decoder](resources/decoder.png)
#### Linear Layer
Converts decoder output to vocabulary probabilities.
#### Transformer
#### Task overview
#### Tokenizer
#### Dataset
#### Training loop
#### Validation loop
#### Attention visualization



**Inspired by:**
[Flash Attention from First Principles (YouTube)](https://www.youtube.com/watch?v=zy8ChVd_oTM)
