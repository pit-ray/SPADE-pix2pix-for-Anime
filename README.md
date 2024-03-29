# SPADE-pix2pix-for-Anime
This pix2pix model is based on Generative Adversarial Networks. I implemented in order to generate from a segmentation label to an anime illust. I could not get a satisfactory result, but obtained enough for small and rough datasets.   
<br>

## Overview
This model generates a anime illust from a segmentation label and a color map of the hair.  
<img src="model_overview.png" />

<br> 

## The differences from the architecture of GauGAN  
- The original `SPADE`[1] computes a weight from label at each itself layers, but this architecture has some mapping networks like the StyleGAN[2] and precompute weights. They are called to `Constant-Resolution FCN` used `AtrousConvolution`[6, 7] instead of `FCN`[5] in order not to do down sampling. In addition, each `SPADE` layer resize and share a encoded weight by a mapping network.  
- I want to assign a hair color, so this Generator has double mapping networks, inputed a segmented label to one mapping network and a RGB color map, to another. The RGB color map has a hair color and positions, it is same resolution with label.  
- I added `NoiseAdder` of StyleGAN[2].  
- I selected a single patch discriminator with `SelfAttention`[3] instead of `Multi-scale discriminator`[1, 10].  
- The discriminator's loss is `Hinge loss` [1] and `Zero Centered Gradient Penalty`[9].  
- The generator's loss is `Hinge loss`, `Feature Matching loss`[1, 10] and `Perceptual loss`[1, 10].  

<br>

## Result  
<img src="https://github.com/pit-ray/SPADE-pix2pix-for-Anime/blob/master/tile.png?raw=true" width=512>  

This result is obtained by training with **500 pixiv images**. This is very small datasets. Additionaly, test datasets is generated by <a href="https://github.com/pit-ray/Anime-Semantic-Segmentation-GAN">Anime-Semantic-Segmentation-GAN</a>, and training datasets is manualy anotated. However, I cannot publish training datasets for copyright of source images.  

The parameters of the upper result is almost same as default value of `options.py`.  
<br>

## pretrained weights  
I prepared pre-trained weights of Generator and Discriminator and added scripts in order to get these weights.  
You can get them by executing a following command.  
```
python get_pretrained_weight.py  
```
Totally about 110MB, so it may take a few minutes.   


## How to predict  
If you want pre-trained model to predict, please do a next python script.   
```
python predict.py  
```
`predict.py` creates predicted images from `predict_from` directory to `predict_to`. Please prepare 256 x 256 png images annotated and 256 x 256 filled hair color. You have to concatenate these.  
  

## How to train
Please create `dataset` directory and prepare dataset. Next, you can set dataset path to option of command.<br>
<br>
```  
Python3 train.py --dataset_dir dataset/example
```  
<br>  

## Environment
||details|
|---|---|
|OS|Windows10 Home|
|CPU|AMD Ryzen 2600|
|GPU|MSI GTX 960 4GB|
|language|Python 3.7.1|
|framework|Chainer 7.0.0, cupy-cuda91 5.3.0|

<br>

## References  
[1] Taesung Park, Ming-Yu Liu, Ting-Chun Wang, Jun-Yan Zhu. Semantic Image Synthesis with Spatially-Adaptive Normalization. <i>arXiv preprint  <a href="https://arxiv.org/abs/1903.07291">arXiv:1903.07291, 2019</a></i>  

[2] Tero Karras, Samuli Laine, Timo Aila. A Style-Based Generator Architecture for Generative Adversarial Networks
. <i>arXiv preprint <a href="https://arxiv.org/abs/1812.04948">arXiv:1812.04948, 2019</a></i>  

[3] Han Zhang, Ian Goodfellow, Dimitris Metaxas, Augustus Odena.  Self-Attention Generative Adversarial Networks. <i>arXiv preprint <a href="https://arxiv.org/abs/1805.08318">arXiv:1805.08318, 2019</a></i>  

[4] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio. Generative Adversarial Networks. <i>arXiv preprint  <a href="https://arxiv.org/abs/1406.2661">arXiv:1406.2661, 2014</a></i>

[5] Jonathan Long, Evan Shelhamer, Trevor Darrell. Fully Convolutional Networks for Semantic Segmentation. <i>arXiv preprint <a href="https://arxiv.org/abs/1411.4038">arXiv:1411.4038, 2015</a></i>

[6] Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, Alan L. Yuille. DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. <i>arXiv preprint <a href="https://arxiv.org/abs/1606.00915">arXiv:1606.00915, 2017 (v2)</a></i>

[7] Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam. Rethinking Atrous Convolution for Semantic Image Segmentation. <i>arXiv preprint <a href="https://arxiv.org/abs/1706.05587">arXiv:1706.05587, 2017 (v3)</a></i>

[8] Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida. Spectral Normalization for Generative Adversarial Networks. <i>arXiv preprint <a href="https://arxiv.org/abs/1802.05957">arXiv:1802.05957, 2018</a></i>

[9] Wenzhe Shi, Jose Caballero, Ferenc Huszár, Johannes Totz, Andrew P. Aitken, Rob Bishop, Daniel Rueckert, Zehan Wang. Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network. <i>arXiv preprint <a href="https://arxiv.org/abs/1609.05158">arXiv:1609.05158, 2016</a></i>

[10] Qi Mao, Hsin-Ying Lee, Hung-Yu Tseng, Siwei Ma, Ming-Hsuan Yang. Mode Seeking Generative Adversarial Networks for Diverse Image Synthesis. <i>arXiv preprint <a href="https://arxiv.org/abs/1903.05628">arXiv:1903.05628, 2019(v6)</a></i>  

[11] Lars Mescheder, Andreas Geiger, Sebastian Nowozin. Which Training Methods for GANs do actually Converge?. <i>arXiv preprint <a href="https://arxiv.org/abs/1801.04406">arXiv:1801.04406, 2018</a></i>  

[12] Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Andrew Tao, Jan Kautz, Bryan Catanzaro. High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs. <i>arXiv preprint <a href="https://arxiv.org/abs/1711.11585">arXiv:1711.11585, 2018</a></i>  

[13] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros. Image-to-Image Translation with Conditional Adversarial Networks. <i>arXiv preprint <a href="https://arxiv.org/abs/1611.07004">arXiv:1611.07004, 2018</a></i>  


## Author  
- pit-ray  
[E-mail] contact(at)pit-ray.com
