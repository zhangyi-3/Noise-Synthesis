# Rethinking Noise Synthesis and Modeling in Raw Denoising (ICCV2021)

[![ICCV 2021](https://img.shields.io/badge/ICCV-2021-blue)](https://openaccess.thecvf.com/content/ICCV2021/html/Zhang_Rethinking_Noise_Synthesis_and_Modeling_in_Raw_Denoising_ICCV_2021_paper.html)
[![arXiv](https://img.shields.io/badge/arXiv-2110.04756-b31b1b)](https://arxiv.org/abs/2110.04756)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.5.0-orange)](https://pytorch.org/)

**Official PyTorch Implementation**

[Yi Zhang](https://zhangyi-3.github.io/)<sup>1</sup>,
[Hongwei Qin](https://scholar.google.com/citations?user=ZGM7HfgAAAAJ&hl=en)<sup>2</sup>,
[Xiaogang Wang](https://scholar.google.com/citations?user=-B5JgjsAAAAJ&hl=zh-CN)<sup>1</sup>,
[Hongsheng Li](https://www.ee.cuhk.edu.hk/~hsli/)<sup>1</sup>

<sup>1</sup>CUHK-SenseTime Joint Lab, <sup>2</sup>SenseTime Research



## Abstract

The lack of large-scale real raw image denoising dataset gives rise to challenges on synthesizing realistic raw image noise for training denoising models. However, the real raw image noise is contributed by many noise sources and varies greatly among different sensors. Existing methods are unable to model all noise sources accurately, and building a noise model for each sensor is also laborious.

**In this paper, we introduce a new perspective to synthesize noise by directly sampling from the sensor's real noise.** It inherently generates accurate raw image noise for different camera sensors. Two efficient and generic techniques: **pattern-aligned patch sampling** and **high-bit reconstruction** help accurate synthesis of spatial-correlated noise and high-bit noise respectively.

We conduct systematic experiments on SIDD and ELD datasets. The results show that:
- Our method **outperforms existing methods** and demonstrates wide generalization on different sensors and lighting conditions
- Recent conclusions derived from DNN-based noise modeling methods are actually based on inaccurate noise parameters. The DNN-based methods still cannot outperform physics-based statistical methods

**📄 Paper:** [arXiv:2110.04756](https://arxiv.org/abs/2110.04756) | [ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Zhang_Rethinking_Noise_Synthesis_and_Modeling_in_Raw_Denoising_ICCV_2021_paper.html)

## 🚀 Getting Started

### Installation

```bash
pip install torch==1.5.0 scikit-image==0.16.2 scipy==1.3.1 h5py==2.10.0 numpy
```

### Quick Example

Generate realistic noise for any camera sensor without requiring the full dataset:

```python
from synthesize import pg_noise_demo
import torch

# Create a sample image
clean_image = torch.randn(256, 256).clamp(0, 1)

# Add realistic noise (supports: 'ip', 's6', 'gp', 'sony', 'nikon')
noisy_image = pg_noise_demo(clean_image, camera='ip')
```

**Supported cameras:** iPhone (`ip`), Samsung S6 (`s6`), Google Pixel (`gp`), Sony a7s2 (`sony`), Nikon D850 (`nikon`)

### Evaluation on SIDD Dataset

1. **Download the dataset:** [SIDD-Medium Dataset](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php)

2. **Download pretrained models:** [Baidu Drive](https://pan.baidu.com/s/1RMYgubpdKZFO8JUkdTC7vg?pwd=v4dh) → place in `checkpoints/` folder

3. **Run evaluation:**
   ```bash
   python -u test.py --root SIDD_Medium/Data --camera s6
   ```
   
   **Available cameras:** `s6` (Samsung S6), `ip` (iPhone), `gp` (Google Pixel)


## 📊 Noise Parameters

The calibrated noise parameters for the SIDD dataset and the ELD dataset can be found in `synthesize.py`.

## 📝 Citation
``` bibtex
 @InProceedings{zhang2021rethinking,
        author    = {Zhang, Yi and Qin, Hongwei and Wang, Xiaogang and Li, Hongsheng},
        title     = {Rethinking Noise Synthesis and Modeling in Raw Denoising},
        booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
        month     = {October},
        year      = {2021},
        pages     = {4593-4601}
    }
```

## 📧 Contact

For questions and inquiries, please contact: **zhangyi@link.cuhk.edu.hk**

## 🙏 Acknowledgments
* [ELD](https://github.com/Vandermode/ELD)
* [CA-NoiseGAN](https://github.com/arcchang1236/CA-NoiseGAN)
* [simple-camera-pipeline](https://github.com/AbdoKamel/simple-camera-pipeline)