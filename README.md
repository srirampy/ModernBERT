# ModernBERT for Zero-Shot Stance Detection

This is the official repository for the paper:  
**“ModernBERT for Zero-Shot Stance Detection: Pairwise Input Representations for Enhanced Performance”**

ModernBERT is designed to be a fast, efficient, and robust model for **Zero-Shot Stance Detection (ZSSD)** across multiple domains.  
The model improves contextual understanding using pairwise input representations while keeping computation lightweight and deployment-friendly.

Our implementation uses:

- **Python 3.10**
- **PyTorch 2.9.1**
- **CUDA 12.6**

Experiments are performed on a single **NVIDIA RTX A5000 GPU**.

---
## Getting Started

1. 10% Training Setting

Run ModernBERT using **10% of the training data**:

```bash
nohup bash ./train_ModernBERT_10_train_tune_tensorboard.sh > train_ModernBERT_10_train_tune_tensorboard.log 2>&1 &
```
2. 100% Training Setting

Run ModernBERT using **full training data**:

```bash
nohup bash ./train_ModernBERT_100_train_tune_tensorboard.sh > train_ModernBERT_100_train_tune_tensorboard.log 2>&1 &
```

## Citation

If you use **ModernBERT** in your research, please cite our paper:

```bash
@INPROCEEDINGS{11257919,
author={Bhavani, Samineni and Kumar, T. Bharath and Manideep, N.V.S. and Sriram, M. Sai and Sathvik, K. Sai and Kommanti, Hima Bindu},
booktitle={2025 International Conference on Emerging Techniques in Computational Intelligence (ICETCI)},
title={ModernBERT for Zero-Shot Stance Detection: Pairwise Input Representations for Enhanced Performance},
year={2025},
pages={1-7},
keywords={Training;Technological innovation;Smoothing methods;Computational modeling;Focusing;Predictive models;Transformers;Natural language processing;Computational intelligence;Context modeling;Zero-Shot Stance Detection;Transformer Models;Natural Language Processing;Label Smoothing},
doi={10.1109/ICETCI67340.2025.11257919}
}
```


