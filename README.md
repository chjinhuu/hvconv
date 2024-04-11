# HVConv

## Abstract
This convolution module is especially designed for **remote sensing** object detection.

## Result
Our model on DOTA-v1.0 obb dataset, baseline method is Oriented R-CNN with R50 and hvconv replaced convolution in R50.
| method | train | batchsize | result | model |
|-------|:-------:|:-------:|:-------:|:-------:|
| baseline | ss | - | 75.87 | - | 
| **hvconv** | ss | 4 | **77.60** | [Google Drive](https://drive.google.com/file/d/1TCIY-aYJT62TuxEkaCI5OQSWtPAQji_m/view?usp=sharing) \| [百度网盘](https://pan.baidu.com/s/1uHoW5sSIEDQ59odXCYN-nQ?pwd=iqar) | 
| baseline | ms+rr | - | 80.87 | - | 
| **hvconv** | ms+rr | 8 | **81.07** | [Google Drive](https://drive.google.com/file/d/13yH2E5b-RLbLPloftRj8Zly2iKbqBGUy/view?usp=sharing) \| [百度网盘](https://pan.baidu.com/s/1gIvfkDYRM5Gp9kh5HukrIw?pwd=4ni8) | 

Our backbone was pretrained **300-epoch** on ImageNet dataset. [Google Drive](https://drive.google.com/file/d/1jN1687_BflfiiIhd3f31NBHJ_bCeoPfB/view?usp=sharing) \| [百度网盘](https://pan.baidu.com/s/1dEz_kzJ9a6_PrtVLcvs1SA?pwd=xnbn)
