# ModernBERT for Zero-Shot Stance Detection  
### Pairwise Input Representations for Enhanced Performance  
**Official Implementation of the IEEE Conference Paper**  
**“ModernBERT for Zero-Shot Stance Detection: Pairwise Input Representations for Enhanced Performance”**

---

## Overview

This repository contains the official implementation of our research work **ModernBERT for Zero-Shot Stance Detection**, accepted at the **2025 IEEE International Conference on Emerging Techniques in Computational Intelligence (ICETCI)**.

Stance detection aims to identify whether a text segment **supports**, **opposes**, or is **neutral** toward a given claim/topic.  
Zero-shot stance detection is challenging because the model has **no task-specific labeled data**.

Our work proposes:

###  **A ModernBERT-based zero-shot stance detector**  
###  **A novel pairwise input representation strategy**  
###  **Better generalization to unseen topics & domains**  

---

## 🚀 Key Contributions

- **Pairwise Input Encoding**  
  We pair the *claim* and *evidence text* using tailored concatenation formats inspired by NLI and semantic similarity tasks.

- **ModernBERT Backbone**  
  Uses the lightweight and efficient **ModernBERT** architecture for faster inference and improved contextual understanding.

- **Zero-Shot Learning Paradigm**  
  The model does **not** require task-specific fine-tuning data and leverages pre-trained knowledge encoded in BERT-style models.

- **Strong Experimental Performance**  
  Achieves performance gains over baselines on multiple stance datasets.

---



