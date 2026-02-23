# ModernBERT for Zero-Shot Stance Detection

> Official PyTorch implementation of the IEEE ICETCI 2025 paper “ModernBERT for Zero-Shot Stance Detection” with pairwise target-text modeling.

This is the **official implementation repository** of our IEEE conference paper:  
**“ModernBERT for Zero-Shot Stance Detection: Pairwise Input Representations for Enhanced Performance”**  
*International Conference on Emerging Techniques in Computational Intelligence (ICETCI 2025)*

📄 Paper Link: https://ieeexplore.ieee.org/document/11257919

---

## Abstract

Zero-Shot Stance Detection (ZSSD) aims to determine whether a given text expresses a **favor**, **against**, or **neutral** stance toward a previously unseen target. Traditional stance detection models often fail when encountering new targets because they learn topic-specific patterns instead of relationships.

In this work, we propose **ModernBERT**, a lightweight Transformer-based architecture that models stance detection as a **pairwise semantic reasoning problem** between the *target* and the *text*. Rather than treating stance detection as a simple classification task, our method explicitly learns the relationship between two inputs.

The proposed model introduces:

- Pairwise input representations  
- Label smoothing for improved generalization  
- Cross-target transfer capability  
- Computationally efficient training  

This repository contains the **complete reproducible implementation** of the method presented in our paper.

---

## Key Idea

Traditional stance detection:

```
Text → Classifier → Stance
```

ModernBERT:

```
(Target, Text) → Transformer Encoder → Semantic Relationship → Stance
```

Instead of memorizing dataset patterns, the model learns:

**“What is the opinion of the text toward the target?”**

---

## Features

- Pairwise target-text modeling  
- Zero-shot generalization to unseen topics  
- Label smoothing regularization  
- Cross-domain transfer capability  
- Robust performance in low-data setting (10% training)  
- Lightweight and deployment-friendly  
- Fully reproducible experiments  

---

## Implementation Details

| Component | Specification |
|----------|-------------|
| Python | 3.10 |
| Framework | PyTorch 2.9.1 |
| CUDA | 12.6 |
| GPU | NVIDIA RTX A5000 |
| Logging | TensorBoard |

---

## Repository Structure

```
ModernBERT/
│
├── models/                # Model architecture
├── utils/                 # Tokenization, metrics, helper functions
├── data/                  # Dataset loaders and preprocessing
├── logs/                  # TensorBoard logs
├── train_ModernBERT_10_train_tune_tensorboard.sh
├── train_ModernBERT_100_train_tune_tensorboard.sh
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/ModernBERT.git
cd ModernBERT
```

Create environment:

```bash
conda create -n modernbert python=3.10
conda activate modernbert
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

### 1. 10% Training Setting (Low-Resource)

Used to evaluate generalization capability:

```bash
nohup bash ./train_ModernBERT_10_train_tune_tensorboard.sh > train_10.log 2>&1 &
```

### 2. 100% Training Setting (Full Training)

Used to reproduce main paper results:

```bash
nohup bash ./train_ModernBERT_100_train_tune_tensorboard.sh > train_100.log 2>&1 &
```

---

## Monitoring Training

Run TensorBoard:

```bash
tensorboard --logdir logs
```

Open in browser:

```
http://localhost:6006
```

You can monitor:

- Training loss  
- Validation loss  
- F1 score  
- Learning rate  

---

## Evaluation

We evaluate using the following metrics:

- Accuracy  
- Macro-F1 Score  
- Weighted-F1 Score  

Zero-shot evaluation is performed by training on **seen targets** and testing on **unseen targets**.

---

## Reproducing Paper Results

To reproduce the results:

1. Use the specified Python and PyTorch versions  
2. Use the provided training scripts  
3. Train using the 100% setting  
4. Evaluate on the zero-shot test split  

All hyperparameters are already included in the scripts.

---

## Applications

- Social media opinion mining  
- Political stance analysis  
- Public opinion monitoring  
- Misinformation detection  
- Cross-domain NLP systems  

---

## Citation

If you use **ModernBERT** in your research, please cite our paper:

```bibtex
@INPROCEEDINGS{11257919,
author={Bhavani, Samineni and Kumar, T. Bharath and Manideep, N.V.S. and Sriram, M. Sai and Sathvik, K. Sai and Kommanti, Hima Bindu},
booktitle={2025 International Conference on Emerging Techniques in Computational Intelligence (ICETCI)},
title={ModernBERT for Zero-Shot Stance Detection: Pairwise Input Representations for Enhanced Performance},
year={2025},
pages={1-7},
doi={10.1109/ICETCI67340.2025.11257919}
}
```

---

## Contribution

Pull requests are welcome.  
For major changes, please open an issue first to discuss.

---

## Acknowledgment

This work was developed as part of our research in **Natural Language Processing and Transformer-based models**.  
We thank the ICETCI reviewers and academic mentors for their valuable feedback.

---

## Contact

**M. Sai Sriram**  
Email: saisriram632@gmail.com  
GitHub: https://github.com/srirampy
