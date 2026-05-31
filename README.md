# Environmental Sound Classification using 1D CNN and CNN-Transformer Hybrid

## Introduction

This project focuses on Environmental Sound Classification using deep learning techniques. The objective is to classify audio recordings into one of 50 sound categories such as dog barking, bird chirping, thunderstorm, vacuum cleaner, crow sounds, can opening sounds, and many other environmental audio events.

Two different architectures were implemented and compared:

* 1D Convolutional Neural Network (1D CNN)
* CNN-Transformer Hybrid Model

The project investigates whether Transformer-based sequence modeling can improve audio classification performance compared to a traditional CNN architecture.

---

# Dataset

## Audio Dataset

* Audio Format: WAV
* Duration: 5 Seconds
* Channels: Mono
* Sample Rate: 44.1 kHz
* Samples per Audio: 220,500
* Number of Classes: 50

Each audio file is associated with:

* File Name
* Category Name
* Category Label

---

# Problem Statement

Given a 5-second environmental sound recording, predict its corresponding category among 50 possible classes.

This is a Multi-Class Classification problem.

---

# Model Comparison

## Final Performance Comparison

| Model             | Accuracy   | F1 Score   | ROC-AUC    |
| ----------------- | ---------- | ---------- | ---------- |
| 1D CNN            | 0.5075     | 0.4931     | 0.9577     |
| CNN + Transformer | 0.45       | 0.44       | 0.94       |

---

# Results

## CNN Training Loss

<p align="center">
<img src="assets/cnn_loss_curve.png" width="900">
</p>

---

## CNN Confusion Matrix

<p align="center">
<img src="assets/cnn_confusion_matrix.png" width="900">
</p>

---

## CNN + Transformer Training Loss

<p align="center">
<img src="assets/transformer_loss_curve.png" width="900">
</p>

---

## CNN + Transformer Confusion Matrix

<p align="center">
<img src="assets/transformer_confusion_matrix.png" width="900">
</p>

---

# Google Colab

🔗 PASTE_COLAB_LINK

---

# Handwritten Report

📄 PASTE_HANDWRITTEN_REPORT_LINK

---

# Trained Models

## 1D CNN

📦 PASTE_CNN_MODEL_LINK

## CNN + Transformer

📦 PASTE_TRANSFORMER_MODEL_LINK

---

# Test Audio Samples

🎵 PASTE_TEST_AUDIO_FOLDER_LINK

---

# Theory

# Audio Preprocessing

Each audio file is loaded using Librosa.

Preprocessing steps:

1. Load WAV file
2. Convert to mono
3. Resample to 44.1 kHz
4. Normalize waveform
5. Convert to tensor format

Input tensor shape:

```text
[B, 1, 220500]
```

where:

* B = Batch Size
* 1 = Mono Channel
* 220500 = Audio Samples

---

# Model 1 : 1D CNN

Unlike image classification which uses 2D convolutions, this project uses 1D convolutions directly on audio waveforms.

---

## CNN Architecture

```text
Input Waveform
[B,1,220500]

        │
        ▼

Conv1D(1→16)
BatchNorm
ReLU
MaxPool

        │
        ▼

Conv1D(16→32)
BatchNorm
ReLU
MaxPool

        │
        ▼

Conv1D(32→64)
BatchNorm
ReLU

        │
        ▼

Conv1D(64→128)
BatchNorm
ReLU

        │
        ▼

Adaptive Average Pool

        │
        ▼

Flatten

        │
        ▼

FC(128→64)

        │
        ▼

Dropout(0.5)

        │
        ▼

FC(64→50)

        │
        ▼

Class Prediction
```

---

## CNN Advantages

* Fast Training
* Lower Computational Cost
* Learns Local Audio Patterns
* Efficient Feature Extraction

---

# Model 2 : CNN + Transformer Hybrid

The Transformer is added on top of CNN-extracted audio features.

CNN learns local patterns while the Transformer captures long-range temporal dependencies.

---

## CNN + Transformer Pipeline

```text
Input Audio

      │
      ▼

1D CNN Feature Extractor

      │
      ▼

Feature Sequence

      │
      ▼

Positional Encoding

      │
      ▼

CLS Token

      │
      ▼

Transformer Encoder

      │
      ▼

CLS Token Output

      │
      ▼

Classification Head

      │
      ▼

50 Classes
```

---

## Transformer Architecture

The Transformer block contains:

```text
LayerNorm
     │
     ▼
Multi Head Attention
     │
Residual Connection
     │
     ▼
LayerNorm
     │
     ▼
Feed Forward Network
     │
Residual Connection
```

---

## Multi-Head Self Attention

The Transformer computes attention using:

$$\text{Attention}(Q,K,V)=\text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
---

## CLS Token

A learnable classification token is added to the sequence.

```text
[CLS]
Audio Feature Tokens
```

After the Transformer encoder, the CLS token representation is used for final classification.

---

## Loss Function

Since this is a multi-class classification problem, Cross Entropy Loss is used.

$$L=-\sum_i y_i\log(\hat{y}_i)$$

where:

- $y_i$ = Ground Truth Label
- $\hat{y}_i$ = Predicted Probability
---

# Training Configuration

| Parameter    | Value     |
| ------------ | --------- |
| Epochs       | 150       |
| Batch Size   | 32        |
| Classes      | 50        |
| Audio Length | 5 Seconds |
| Sample Rate  | 44.1 kHz  |

---

# Key Observations

* CNN successfully learns local waveform patterns.
* Transformer captures long-range temporal dependencies.
* The CNN-Transformer hybrid improves contextual understanding of audio signals.
* Multi-head attention allows the model to focus on important temporal regions.
* Combining CNN and Transformer provides a powerful architecture for environmental sound classification.

---


# Conclusion

This project demonstrates environmental sound classification using both convolutional and transformer-based architectures. A traditional 1D CNN serves as the baseline model, while a CNN-Transformer hybrid extends the architecture by incorporating self-attention mechanisms for sequence modeling. The comparative analysis highlights the effectiveness of transformer-based approaches in learning long-range audio dependencies.

---

## Author

**Pranav Deshpande**
IIT Jodhpur

* Deep Learning 
* Audio Processing
* Transformers
* Computer Vision
* Generative AI

