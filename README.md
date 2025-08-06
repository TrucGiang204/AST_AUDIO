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
  <img src="./image/fig_0216.png" alt="ADD4RSC model architecture" width="600"/>
</p>


## 🚀 Getting Started



### 1. 📦 Download and prepare audio data
- **Download**: ICBHI 2017 Respiratory Sound Database [Official Challenge Page](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge)
- **Characteristics**:
  - 126 subjects with real-world recordings
  - Contains heart sounds, ambient noise, and transducer artifacts
  - Sampling rates: 4kHz-44.1kHz → resampled to 16kHz


### 2. 🤖 Download pre-trained model
You can download the pretrained AST model from [Hugging Face](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593).


### 3. 🧠 Start training
Our code is based on PyTorch. Please install these required packages from their official sources. We include our versions below for reference, but other versions might also work.

```
# Main packages for training
pytorch=2.0.1
cuda=11.7
torchaudio=2.0.1
```

### 4. 📊 Comparison of Our Method with State-of-the-Art on the ICBHI Dataset

| Model                         | $S_p$(%) | $S_e$(%) | Score(%) | Publication   |
|------------------------------|----------|----------|----------|----------------|
| LungRN+NL [25]               | 63.20    | 41.32    | 52.26    | Interspeech    |
| RespireNet [3]               | 72.30    | 40.10    | 56.20    | Interspeech    |
| Wang *et al.* [7] (Splice)   | 70.40    | 40.20    | 55.30    | Interspeech    |
| StochNorm [26]               | 78.86    | 36.40    | 57.63    | Interspeech    |
| CoTuning [26]                | 79.34    | 37.24    | 58.29    | Interspeech    |
| Chang *et al.* [27]          | 69.92    | 35.85    | 52.89    | Interspeech    |
| SCL [28]                     | 75.95    | 39.15    | 57.55    | Interspeech    |
| **Ours (ResNet50)**          | **83.76**| **34.18**| **58.97**| Interspeech    |
|------------------------------|----------|----------|----------|----------------|
| AFT on Mixed-500 [5]         | 80.72    | 42.86    | 61.79    | Interspeech    |
| AST Fine-tuning [9]          | 77.14    | 41.97    | 59.55    | Interspeech    |
| Patch-Mix CL [9]             | 81.66    | 43.07    | 62.37    | Interspeech    |
| M2D [29]                     | 81.51    | 45.08    | 63.29    | Interspeech    |
| DAT [8]                      | 77.11    | 42.50    | 59.81    | Interspeech    |
| SG-SCL [8]                   | 79.87    | 43.55    | 61.71    | Interspeech    |
| RepAugment [4]               | 82.47    | 40.55    | 61.51    | Interspeech    |
| BTS [30]                     | 81.40    | 45.67    | 63.54    | Interspeech    |
| MVST [31]                    | 80.60    | 44.39    | 62.50    | Interspeech    |
| LungAdapter [32]            | 80.43    | 44.37    | 62.40    | Interspeech    |
| CycleGuardian [33]           | 82.06    | 44.47    | 63.26    | Interspeech    |
| **Ours (AST)**               | **85.13**| **45.94**| **65.53**| Interspeech    |
|------------------------------|----------|----------|----------|----------------|


## 📚 Citation

Please cite related papers if you use ADD4RSC.

```
@article{dong2025adaptive,
  title={Adaptive Differential Denoising for Respiratory Sounds Classification},
  author={Dong, Gaoyang and Zhang, Zhicheng and Sun, Ping and Zhang, Minghui},
  journal={arXiv preprint arXiv:2506.02505},
  year={2025}
}
```

## Acknowledgments

