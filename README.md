# Eliminating Moiré Patterns Across Diverse Image Resolutions via DMMNet (IEEE TMM, 2025)

# Introduction

This repository contains the official PyTorch implementation for the TMM 2025 paper titled "Eliminating Moiré Patterns Across Diverse Image Resolutions via DMMNet" by Yikun Ma, Haoran Qi, and Zhi Jin.

# Demo

![81fe4a871a7af24dc0782921d779450](https://github.com/user-attachments/assets/7aa6e2e9-c0a5-4681-a3cc-530111864dd6)


# Abstract
The occurrence of frequency aliasing between the camera and high-frequency scene elements causes moire patterns in images, leading to color distortions and a loss of fine details, thereby reducing image quality. The intricate frequency characteristics and diverse appearances inherent in moire patterns render their removal, commonly referred to as demoireing, particularly challenging. Recent advancements in deep learningbased demoireing methods have showcased notable efficacy. However, prevailing techniques often specialize in mitigating moire patterns exclusively within either the frequency or spatial domains. Additionally, these methods generally perform well at specific image resolutions, but struggle to maintain effectiveness across different resolutions due to less generalized architectures. To address these issues, we propose a Dual-domain Multilevel Multi-scale Network DMMNet, working in both spatial and frequency domains sequentially. The Multi-scale Multi-level Demoire Stage (MMDS) in our framework focuses on moire patterns removal in the spatial domain. To adeptly integrate
features from various semantic levels, we introduce a pioneering plug-and-play Adjacent Cross Attention (ACA) module within the MMDS. Subsequently, the Frequency Separation and Reconstruction Stage (FSRS) restores high-frequency texture details, reconstructs color information, and eliminates residual moire patterns in the wavelet frequency domain. Ultimately, the clean image is obtained by converting it back to the spatial domain. Extensive experimental assessments, spanning both quantitative metrics and qualitative visual evaluations, attest to the superior efficacy of DMMNet to State-Of-The-Art (SOTA) demoireing methods, concurrently exhibiting enhanced generalization for demoireing across diverse image resolutions. We posit that the proposed methodology presents a viable solution for broader applications in the realm of demoireing.

# Getting Started

## 1. Installation
Our env is cuda-12.0.
```
pip install -r requirements.txt
```

## 2. Running:
You are free to choose to train our DMMNet on a specific dataset. If you want to improve the generalization across different resolutions, we recommend mixing different datasets.
Moreover, in subsequent experiments, we found that using cosine distance as the color loss also yields good results (refer to the loss_util.py). 

```
#Train mmds in the first stage
python train_mmds_waca.py --config ./config/mmds.yaml

#Then train fsrs 
python train_fsrs.py --config ./config/fsrs.yaml
```

# Acknowledgement
This work was supported by [Frontier Vision Lab](https://fvl2020.github.io/fvl.github.com/), SUN YAT-SEN University. 
We would like to thank the authors of [UHDM](https://github.com/CVMI-Lab/UHDM) for their great work and generously providing source codes, 
which inspired our work and helped us a lot in the implementation.

# Citation
If you find this work helpful, please consider citing:

```
@article{ma2025eliminating,
  title={Eliminating Moire Patterns Across Diverse Image Resolutions via DMMNet},
  author={Ma, Yikun and Qi, Haoran and Jin, Zhi},
  journal={IEEE Transactions on Multimedia},
  year={2025},
  publisher={IEEE},
  pages={1-11},
  doi={10.1109/TMM.2025.3535324}
}
```
