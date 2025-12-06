# ModernBERT for Zero-Shot Stance Detection

This is the official repository for the paper:  
**“ModernBERT: A Lightweight and Efficient Transformer Model for Zero-Shot Stance Detection.”**

ModernBERT is designed to be a fast, efficient, and robust model for **Zero-Shot Stance Detection (ZSSD)** across multiple domains.  
The model improves contextual understanding using pairwise input representations while keeping computation lightweight and deployment-friendly.

Our implementation uses:

- **Python 3.8+**
- **PyTorch 1.10+**
- **CUDA 11.x**

Experiments are performed on a single **NVIDIA RTX A5000 GPU**.

---
## 10% Training Setting

Run ModernBERT using **10% of the training data**:

```bash
nohup bash ./train_ModernBERT_10_train_tune_tensorboard.sh > train_ModernBERT_10_train_tune_tensorboard.log 2>&1 &

## 100% Training Setting

Run ModernBERT using **full training data**:

'''bash
nohup bash ./train_ModernBERT_100_train_tune_tensorboard.sh > train_ModernBERT_100_train_tune_tensorboard.log 2>&1 &



