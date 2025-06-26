# ADD-RSC

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

Please cite related papers if you use DPHuBERT.

```
@inproceedings{peng23c_interspeech,
  author={Yifan Peng and Yui Sudo and Shakeel Muhammad and Shinji Watanabe},
  title={{DPHuBERT: Joint Distillation and Pruning of Self-Supervised Speech Models}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
  pages={62--66},
  doi={10.21437/Interspeech.2023-1213}
}

```

## Acknowledgments

