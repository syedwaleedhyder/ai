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
#### Layer Normalization
#### Feed Forward
#### Multi-Head Attention
#### Residual Connection
#### Encoder
#### Decoder
#### Linear Layer
#### Transformer
#### Task overview
#### Tokenizer
#### Dataset
#### Training loop
#### Validation loop
#### Attention visualization



**Inspired by:**
[Flash Attention from First Principles (YouTube)](https://www.youtube.com/watch?v=zy8ChVd_oTM)
