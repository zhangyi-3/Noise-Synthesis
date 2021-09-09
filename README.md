Rethinking Noise Synthesis and Modeling in Raw Denoising (ICCV2021)
---
[Yi Zhang](https://zhangyi-3.github.io/)<sup>1</sup>,
[Hongwei Qin](https://scholar.google.com/citations?user=ZGM7HfgAAAAJ&hl=en)<sup>2</sup>,
[Xiaogang Wang](https://scholar.google.com/citations?user=-B5JgjsAAAAJ&hl=zh-CN)<sup>1</sup>,
[Hongsheng Li](https://www.ee.cuhk.edu.hk/~hsli/)<sup>1</sup><br>

<sup>1</sup>CUHK-SenseTime Joint Lab, <sup>2</sup>SenseTime Research



### Abstract

>The lack of large-scale real raw image denoising dataset gives rise to challenges on synthesizing 
realistic raw image noise for training denoising models.However, the real raw image noise is 
contributed by many noise sources and varies greatly among different sensors.
Existing methods are unable to model all noise sources accurately, and building a noise model 
for each sensor is also laborious. In this paper, we introduce a new perspective to synthesize 
noise by directly sampling from the sensor's real noise.It inherently generates accurate raw image 
noise for different camera sensors. Two efficient and generic techniques: pattern-aligned patch 
sampling and high-bit reconstruction help accurate synthesis of spatial-correlated noise and high-bit noise respectively. We conduct systematic experiments on SIDD and ELD datasets. 
The results show that  (1) our method outperforms existing methods and demonstrates wide 
generalization on different sensors and lighting conditions. (2) Recent conclusions derived from 
DNN-based noise modeling methods are actually based on inaccurate noise parameters. 
The DNN-based methods still cannot outperform physics-based statistical methods.

### Code & calibrated parameters

coming soon ！