# ADD-RSC(Refactoring the codebase)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/adaptive-differential-denoising-for/audio-classification-on-icbhi-respiratory)](https://paperswithcode.com/sota/audio-classification-on-icbhi-respiratory?p=adaptive-differential-denoising-for)
[![Paper](https://img.shields.io/badge/arXiv-2506.02505-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2506.02505)
[![Model License](https://img.shields.io/badge/Model_License-Apache_2.0-olive)](https://opensource.org/licenses/Apache-2.0)

This repo contains the code and models for our paper: 

Dong Gaoyang, Zhang Zhicheng, Sun Ping and Zhang Minghui, "Adaptive Differential Denoising for Respiratory Sounds Classification", and accepted at Interspeech 2025.
[[arXiv](https://arxiv.org/pdf/2506.02505)]


## Overview




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

