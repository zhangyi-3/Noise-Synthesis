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

### Testing
The code has been tested with the following environment:
```
pytorch == 1.5.0
scikit-image == 0.16.2
scipy == 1.3.1
h5py 2.10.0 
```
    
- Prepare the [SIDD-Medium Dataset](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php) dataset. 
- Download the [pretrained models](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155135732_link_cuhk_edu_hk/Egb3x2YO-qBBgQ41N8WiCIUBRQuxb4gWsV_Ml1yLfDti9w?e=wRwC7e) and put them into the checkpoints folder.
- Modify the default root of the SIDD dataset and run the command with the specific camera name (s6 | ip | gp) 
```
python -u test.py --root SIDD_Medium/Data --camera s6
```


### Noise parameters
The calibrated noise parameters for the SIDD dataset and the ELD dataset can be found in ``synthesize.py``.


### Citation
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

### Contact
Feel free to contact zhangyi@link.cuhk.edu.hk if you have any questions.

### Acknowledgments
* [ELD](https://github.com/Vandermode/ELD)
* [CA-NoiseGAN](https://github.com/arcchang1236/CA-NoiseGAN)
* [simple-camera-pipeline](https://github.com/AbdoKamel/simple-camera-pipeline)