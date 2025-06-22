# tiny-llm Learning Project

> This is a fork of [skyzh/tiny-llm](https://github.com/skyzh/tiny-llm) for personal learning

## Project Overview

This is my forked learning project based on tiny-llm, which is a course on LLM serving using MLX framework to build large language model serving infrastructure from scratch.

**Learning Goals**:
- Deep understanding of large language model implementation principles
- Master efficient LLM serving techniques
- Learn MLX framework usage
- Understand Transformer architecture components

**Tech Stack**:
- **MLX**: Apple's machine learning framework optimized for Apple Silicon
- **Python**: Primary programming language
- **Qwen2**: Target model architecture

## Original Project Information

- **Original Author**: [skyzh](https://github.com/skyzh)
- **Original Repository**: [https://github.com/skyzh/tiny-llm](https://github.com/skyzh/tiny-llm)
- **Project Documentation**: [https://skyzh.github.io/tiny-llm/](https://skyzh.github.io/tiny-llm/)

## My Learning Progress

### Week 1: Basic Model Implementation
- [x] **1.1 Attention Mechanism**: Understand and implement scaled dot-product attention
- [ ] **1.2 Positional Encoding**: Implement RoPE (Rotary Position Embedding)
- [ ] **1.3 Grouped Query Attention**: Learn GQA optimization techniques
- [ ] **1.4 Normalization and MLP**: Implement RMSNorm and feed-forward networks
- [ ] **1.5 Model Loading**: Learn model weight loading and management
- [ ] **1.6 Response Generation**: Implement decoding process
- [ ] **1.7 Sampling Strategies**: Implement various sampling methods

### Week 2: Performance Optimization
- [ ] **2.1 KV Cache**: Implement key-value cache optimization
- [ ] **2.2 Quantized MatMul (CPU)**: CPU-side quantization optimization
- [ ] **2.3 Quantized MatMul (GPU)**: GPU-side quantization optimization
- [ ] **2.4 Flash Attention (CPU)**: CPU-side attention optimization
- [ ] **2.5 Flash Attention (GPU)**: GPU-side attention optimization
- [ ] **2.6 Continuous Batching**: Batching optimization techniques
- [ ] **2.7 Chunked Prefill**: Prefill optimization strategies

### Week 3: Advanced Techniques
- [ ] **3.1-3.2 Paged Attention**: Memory management optimization
- [ ] **3.3 Mixture of Experts (MoE)**: Sparse model techniques
- [ ] **3.4 Speculative Decoding**: Inference acceleration techniques
- [ ] **3.5 Prefill-Decode Separation**: Architecture optimization
- [ ] **3.6 Parallelization**: Distributed inference
- [ ] **3.7 AI Agents**: Tool calling implementation

## Learning Resources

- [Original Project Documentation](https://skyzh.github.io/tiny-llm/)
- [MLX Official Documentation](https://ml-explore.github.io/mlx/build/html/index.html)
- [Transformer Paper](https://arxiv.org/abs/1706.03762)
- [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671)

---

**Note**: This is a personal learning project for deep understanding of large language model implementation details. If you're interested in the original project, please visit [skyzh/tiny-llm](https://github.com/skyzh/tiny-llm).
