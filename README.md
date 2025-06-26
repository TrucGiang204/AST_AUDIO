# ADD-RSC(Refactoring the codebase)
[![Paper](https://img.shields.io/badge/arXiv-2506.02505-red.svg?style=flat)](https://arxiv.org/abs/2506.02505)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/adaptive-differential-denoising-for/audio-classification-on-icbhi-respiratory)](https://paperswithcode.com/sota/audio-classification-on-icbhi-respiratory?p=adaptive-differential-denoising-for)

[![Model License](https://img.shields.io/badge/Model_License-Apache_2.0-olive)](https://opensource.org/licenses/Apache-2.0)

This repo contains the code and models for our paper: 

Dong Gaoyang, Zhang Zhicheng, Sun Ping and Zhang Minghui, "Adaptive Differential Denoising for Respiratory Sounds Classification", and accepted at Interspeech 2025.
[[arXiv](https://arxiv.org/pdf/2506.02505)]


## Overview
This repository contains the official implementation of the **Adaptive Differential Denoising (ADD)** network for robust respiratory sound classification. The method integrates three key innovations:

1. **Adaptive Frequency Filter (AFF)**  
   - Learns spectral masks with soft shrink: `Sα(x) = sign(x) max{|x| - α, 0}`
   - Preserves diagnostic high-frequency components while eliminating noise

2. **Differential Denoise Layer (DDL)**  
   - Uses Multi-Head Differential Attention (MHDA) mechanism
   - Suppresses noise through contrastive augmented views
   - Implements:  
     `A = [σ(Q₁K₁ᵀ/√d) - λ·σ(Q₂K₂ᵀ/√d)]V`

3. **Bias Denoising Loss**  
   - Jointly optimizes classification and denoising
   - Uses label smoothing: `L = -Σ[y_c(1-ε) + ε/C]·log[ϕ(Norm(p))]`
   - No clean labels required

**Key Results**:
- 🏆 **65.53% Score** on ICBHI 2017 (SOTA)
- 📈 **+1.99%** improvement over previous best
- 🔍 **85.13% Specificity (Sp)** | **45.94% Sensitivity (Se)**

## Data
### ICBHI 2017 Respiratory Sound Database
- **Download**: [Official Challenge Page](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge)
- **Characteristics**:
  - 126 subjects with real-world recordings
  - Contains heart sounds, ambient noise, and transducer artifacts
  - Sampling rates: 4kHz-44.1kHz → resampled to 16kHz



## Requirements

Our code is based on PyTorch, TorchAudio, and PyTorch Lightning. Please install these required packages from their official sources. We include our versions below for reference, but other versions might also work.

```
# Main packages for training
pytorch=1.13.1
cuda=11.6.2
pytorch-lightning=1.8.1
torchaudio=0.13.1

# Other packages for obtaining pre-trained SSL
fairseq=0.12.2
transformers=4.24.0
```


## Usage



### 1. Download and prepare audio data



### 2. Download pre-trained SSL (e.g., HuBERT Base) and convert it to our format



### 3. Start training




## Pre-trained models




## Citation

Please cite related papers if you use ADD-RSC.

```
@article{dong2025adaptive,
  title={Adaptive Differential Denoising for Respiratory Sounds Classification},
  author={Dong, Gaoyang and Zhang, Zhicheng and Sun, Ping and Zhang, Minghui},
  journal={arXiv preprint arXiv:2506.02505},
  year={2025}
}

```

## Acknowledgments

