# ADD4RSC(Refactoring the codebase)
[![Paper](https://img.shields.io/badge/arXiv-2506.02505-red.svg?style=flat)](https://arxiv.org/abs/2506.02505)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/adaptive-differential-denoising-for/audio-classification-on-icbhi-respiratory)](https://paperswithcode.com/sota/audio-classification-on-icbhi-respiratory?p=adaptive-differential-denoising-for)

[![Model License](https://img.shields.io/badge/Model_License-Apache_2.0-olive)](https://opensource.org/licenses/Apache-2.0)

This repo contains the code and models for our paper: 

Dong Gaoyang, Zhang Zhicheng, Sun Ping and Zhang Minghui, "Adaptive Differential Denoising for Respiratory Sounds Classification", and accepted at Interspeech 2025.
[[arXiv](https://arxiv.org/pdf/2506.02505)]


## Overview
Automated respiratory sound classification faces practical challenges from background noise and insufficient denoising in existing systems. We propose **A**daptive **D**ifferential **D**enoising network for **R**espiratory **S**ounds **C**lassification (**ADD4RSC**), that integrates noise suppression and pathological feature preservation via three innovations: 1) **Adaptive Frequency Filter** with learnable spectral masks and soft shrink to eliminate noise while retaining diagnostic high-frequency components; 2) A **Differential Denoise Layer** using differential attention to reduce noise-induced variations through augmented sample comparisons; 3) A **bias denoising loss** jointly optimizing classification and robustness without clean labels. Experiments on the ICBHI2017 dataset show that our method achieves `65.53%` of the Score, which is improved by `1.99%` over the previous sota method.

<p align="center">
  <img src="fig_0216.png" alt="ADD4RSC model architecture" width="600"/>
</p>

## Data
### ICBHI 2017 Respiratory Sound Database
- **Download**: [Official Challenge Page](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge)
- **Characteristics**:
  - 126 subjects with real-world recordings
  - Contains heart sounds, ambient noise, and transducer artifacts
  - Sampling rates: 4kHz-44.1kHz → resampled to 16kHz



## Requirements

Our code is based on PyTorch. Please install these required packages from their official sources. We include our versions below for reference, but other versions might also work.

```
# Main packages for training
pytorch=2.0.1
cuda=11.7
torchaudio=2.0.1

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

