---
layout: post
title: "Prominent ConvNet Architectures"
permalink: /convnet-architectures/
mathjax: true
---


<style>
  table{border-spacing: 0px 0px;}
  tr:nth-child(odd){background:#ccc;}
  th{background:white;padding-left:10px;padding-right:10px}
  td{padding-left:15px;padding-right:15px}
</style>

## ResNet

**Paper**: <br />
[Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385) (K. He, X. Zhang, S. Ren and J. Sun; 10 Dec 2015) <br /> 

**Code:** <br />
[Original implementation by K. He](https://github.com/KaimingHe/deep-residual-networkst) (models are converted to a recent (2016/2/3) version of Caffe) and [Torch implementation by Facebook AI Research](https://github.com/facebook/fb.resnet.torch) <br /> 

**Description:** <br />
Winner of the [ILSVRC 2015](http://image-net.org/challenges/LSVRC/2015/) object detection and image classification and localization tasks. Neural networks with depth of over 150 layers are used together with a "deep residual learning" framework that eases the optimization and convergence of extremely deep networks. The localization and detection systems are in addition based on the ["Faster R-CNN"](http://arxiv.org/abs/1506.01497) system of S. Ren at al


| Layer Name | Output Size | 18-Layer Net | 34-Layer Net | 50-Layer Net | 101-Layer Net | 152-Layer Net |
|:---------- | ----------- | ------------ | ------------ | ------------ | ------------- | ------------- |
| conv1      | 112×112     |              | 7x7, 64, stride 2                                           |               
| conv2_x    | 56x56       |              | 3x3 max pool, stride 2                                      |
| conv2_x    | 56x56       | \\( \left[ \overset{3x3, 64}{3x3, 64} \right] \times 2 \\)  |     |     |    |     |


<br /> 

## GoogLeNet

**Paper**: <br />
[Going Deeper with Convolutions](http://arxiv.org/abs/1409.4842) (C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke and A. Rabinovich; 17 Sep 2014) <br /> 

**Code:** <br />
[Caffe-Model](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet) and [GoogLeNet GPU implementation from Princeton](http://vision.princeton.edu/pvt/GoogLeNet/) (\\(\rightarrow\\) see "Trained model using ImageNet") (from [Caffe Model-Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)) <br /> 

**Description:** <br />
Winner of the [ILSVRC 2014](http://www.image-net.org/challenges/LSVRC/2014/) at the categories *"Classification"* (with provided training data) and *"Detection"* (with additional training data). The main innovation of GoogLeNet is the *"Inception Module"* that significantly reduces the number of network parameters. The *"Inception Module"* is an aggregation of optimal local sparse (sparsely connected) network structures into readily available building blocks. The architecture of the *"Inception Module"* and the overall network architecture are shown in the subsequent tables. "All convolutions, including those inside the Inception modules, use rectified linear activation (ReLU). The size of the receptive field is 224×224 taking RGB color channels with mean subtraction. “#3×3 reduce” and “#5×5 reduce” stands for the number of 1×1 filters in the reduction layer used before the 3×3 and 5×5 convolutions. One can see the number of 1×1 filters in the projection layer after the built-in max-pooling in the pool proj column. All these reduction/projection layers use rectified linear activation (ReLU) as well." (Going Deeper with Convolutions, C. Szegedy et al, 2014)


|           | Inception Module     |           |                 |
| :-------- | :------------------: | :-------- | :-------------: |
|           | Previous layer       |           |                 |
|  conv 1x1 | conv 1x1             | conv 1x1  | max pooling 3x3 |
|           | conv 3x3             | conv 5x5  | conv 1x1        |
|           | Filter Concatenation |           |                 |

<br /> 

| Type           | Size/Stride | Output Size | Depth | #1×1 | #3×3 reduce | #3×3 | #5×5 reduce | #5×5 | pool proj |
|:-------------- | ------------------- | ----------- | ----- | ---- | ----------- | ---- | ----------- | ---- | --------- |
| convolution    | 7×7 /2              | 112×112×64  | 1     |      |             |      |             |      |           |
| max pool       | 3×3 /2              | 56×56×64    | 0     |      |             |      |             |      |           |
| convolution    | 3×3 /1              | 56×56×192   | 2     |      | 64          | 192  |             |      |           |
| max pool       | 3×3 /2              | 28×28×192   | 0     |      |             |      |             |      |           |
| inception(3a)  |                     | 28×28×256   | 2     | 64   | 96          | 128  | 16          | 32   | 32        |
| inception(3b)  |                     | 28×28×480   | 2     | 128  | 128         | 192  | 32          | 96   | 64        |
| max pool       | 3×3 /2              | 14×14×480   | 0     |      |             |      |             |      |           |
| inception(4a)  |                     | 14×14×512   | 2     | 192  | 96          | 208  | 16          | 48   | 64        |
| inception(4b)  |                     | 14×14×512   | 2     | 160  | 112         | 224  | 24          | 64   | 64        |
| inception(4c)  |                     | 14×14×512   | 2     | 128  | 128         | 256  | 24          | 64   | 64        |
| inception(4d)  |                     | 14×14×528   | 2     | 112  | 144         | 288  | 32          | 64   | 64        |
| inception(4e)  |                     | 14×14×832   | 2     | 256  | 160         | 320  | 32          | 128  | 128       |
| max pool       | 3×3 /2              | 7×7×832     | 0     |      |             |      |             |      |           |
| inception(5a)  |                     | 7×7×832     | 2     | 256  | 160         | 320  | 32          | 128  | 128       |
| inception(5b)  |                     | 7×7×1024    | 2     | 384  | 192         | 384  | 48          | 128  | 128       |
| avg pool       | 7×7 /1              | 1×1×1024    | 0     |      |             |      |             |      |           |
| dropout(40%)   |                     | 1×1×1024    | 0     |      |             |      |             |      |           |
| linear         |                     | 1×1×1000    | 1     |      |             |      |             |      |           |
| softmax        |                     | 1×1×1000    | 0     |      |             |      |             |      |           |

<br /> 

## OxfordNet (also VGGNet)

**Paper**: <br />
[Very Deep Convolutional Networks for Large-Scale Image Recognition](http://arxiv.org/abs/1409.1556) (K. Simonyan and A. Zisserman; 4 Sep 2014) <br /> 

**Code**: <br />
[Caffe-Model](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md) for network configuration D from the paper (see also [Caffe Model-Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo) for other network configurations); [Darknet-Model](https://github.com/pjreddie/darknet/blob/master/cfg/vgg-16.cfg); [Lasagne-Model](https://github.com/Lasagne/Recipes/blob/master/modelzoo/vgg16.py) <br /> 

**Description**: <br />
Winner of the [ILSVRC 2014](http://www.image-net.org/challenges/LSVRC/2014/) at the categories *"Localization"* (with provided training data) and second place at the category *"Classification"*. The network architecture (**configuration D**) for handling the **ILSVRC-2014** dataset is shown in the subsequent table. Each convolutional and fully connected layer (except for the last) are followed by a rectifier linear unit (**ReLU**) activation function.

| Input Size \\(\hspace{0.5cm}\\) | VGG16 |
|:-------------- |:---------------------- |
|  224 x 224     | conv 3x3, 64           |
|  224 × 224     | conv 3x3, 64           |
|  224 × 224     | max pooling 2x2 /2     |
|  112 × 112     | conv 3x3, 128          |
|  112 × 112     | conv 3x3, 128          |
|  112 × 112     | max pooling 2x2 /2     |
|   56 × 56      | conv 3x3, 256          |
|   56 × 56      | conv 3x3, 256          |
|   56 × 56      | conv 3x3, 256          |
|   56 × 56      | max pooling 2x2 /2     |
|   28 × 28      | conv 3x3, 512          |
|   28 × 28      | conv 3x3, 512          |
|   28 × 28      | conv 3x3, 512          |
|   28 × 28      | max pooling 2x2 /2     |
|   14 × 14      | conv 3x3, 512          |
|   14 × 14      | conv 3x3, 512          |
|   14 × 14      | conv 3x3, 512          |
|   14 × 14      | max pooling 2x2 /2     |
| 7 × 7 × 512    | fc, 4096               |
| 4096 × 1       | dropout 0.5            |
| 4096 × 1       | fc, 4096               |
| 4096 × 1       | dropout 0.5            |
| 4096 × 1       | fc, 1000               |
| 1000           | softmax                |

<br /> 

## ZFNet

**Paper**: <br />
[Visualizing and Understanding Convolutional Networks](http://arxiv.org/abs/1311.2901) (M. D. Zeiler and R. Fergus; 12 Nov 2013) <br /> 

**Code**: <br />
[Caffe-Model](https://github.com/BVLC/caffe/blob/47855ca9596615b65aeebbeeac72ff78aca0c5e3/examples/imagenet-winner2013.prototxt) (old version; not included in Caffe any more) (see also this [discussion about the Caffe-Model](https://github.com/BVLC/caffe/pull/33)) (New reference model for ImageNet training from Berkeley Vision is: [bvlc_reference_caffenet](https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet).) <br /> 

**Description**: <br />
Winner of the [ILSVRC 2013](http://image-net.org/challenges/LSVRC/2013/) at the category *"Classification"*. The network architecture described in the paper is shown in the subsequent table. Each convolutional and fully connected layer are followed by a rectifier linear unit (**ReLU**) activation function. Note, the network architecture is a modification of AlexNet (see below) that outperforms their 2012 results on ImageNet. E.g. the 1st layer filter size is reduced from 11x11 to 7x7 and the stride of the convolution is reduced from 4 to 2. This modification retains
more information in the 1st and 2nd layer features and thus improves classification performance. (lrn = local response normalization; see Sect. 3.3 of the AlexNet paper below)

| Input Size \\(\hspace{0.5cm}\\) | ZFNet      |
|:-------------- |:--------------------------- |
| 224 × 224      | conv 7x7 /2, 96             | 
| 110 × 110      | max pooling 3x3 /2          |
| 55 × 55        | lrn (n=5, α=0.0001, β=0.75) |
| 55 × 55        | conv 5x5 /2, 256            |
| 26 × 26        | max pooling 3x3 /2          |
| 13 × 13        | lrn (n=5, α=0.0001, β=0.75) |
| 13 × 13        | conv 3x3 /1, 384            |
| 13 × 13        | conv 3x3 /1, 384            |
| 13 × 13        | conv 3x3 /1, 256            |
| 13 × 13        | max pooling 3x3 /2          |
| 6 × 6 × 256    | fc, 4096                    |
| 4096 × 1       | dropout 0.5                 |
| 4096 × 1       | fc, 4096                    |
| 4096 × 1       | dropout 0.5                 |
| 1000           | softmax                     |

<br /> 

## AlexNet

**Paper**: <br />
[ImageNet Classification with Deep Convolutional Neural Networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) (A. Krizhevsky, I. Sutskever and G. E. Hinton; 2012) <br />

**Code**: <br />
[Caffe-Model](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet) <br />

**Description**: <br />
Winner of the [ILSVRC 2012](http://image-net.org/challenges/LSVRC/2012/) at the categories *"Classification"* and *"Localization"*. The network architecture described in the paper is shown in the subsequent table. A **ReLU** non-linearity is applied to the output of every convolutional and fully-connected layer. (lrn = local response normalization; see Sect. 3.3 of the paper)

| Input Size \\(\hspace{0.5cm}\\) | AlexNet    |
|:-------------- |:--------------------------- |
| 224 × 224      | conv 11x11 /4, 96           | 
| 55 × 55        | lrn (n=5, α=0.0001, β=0.75) |
| 55 × 55        | max pooling 3x3 /2          |
| 27 × 27        | conv 5x5 /1, 256            |
| 27 × 27        | lrn (n=5, α=0.0001, β=0.75) |
| 27 × 27        | max pooling 3x3 /2          |
| 13 × 13        | conv 3x3 /1, 384            |
| 13 × 13        | conv 3x3 /1, 384            |
| 13 × 13        | conv 3x3 /1, 256            |
| 13 × 13        | max pooling 3x3 /2          |
| 6 × 6 × 256    | fc, 4096                    |
| 4096 × 1       | dropout 0.5                 |
| 4096 × 1       | fc, 4096                    |
| 4096 × 1       | dropout 0.5                 |
| 1000           | softmax                     |

<br /> 

## NIN (Network in Network)

**Paper**: <br />
[Network In Network](http://arxiv.org/pdf/1312.4400v3.pdf) (M. Lin, Q. Chen and S. Yan; 16 Dec 2013) <br /> 

**Code**: <br />
[Caffe-Model](https://gist.github.com/mavenlin/e56253735ef32c3c296d) (from [Caffe Model-Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)) <br /> 

**Description:** <br />
The network architecture for handling the **CIFAR-10** and **CIFAR-100** dataset, respectively, is shown in the subsequent table. Each convolutional layer is followed by a rectifier linear unit (**ReLU**) activation function.

| Input Size \\(\hspace{0.5cm}\\) | NiN   |
|:-------------- |:---------------------- |
| 32 x 32        | conv 5x5, 192          | 
| 32 × 32        | conv 1x1, 160          |
| 32 × 32        | conv 1x1, 96           |
| 32 × 32        | max pooling 3x3 /2     |
| 16 × 16        | dropout 0.5            |
| 16 × 16        | conv 5x5, 192          |
| 16 × 16        | conv 1x1, 192          |
| 16 × 16        | conv 1x1, 192          |
| 16 × 16        | avg pooling 3x3 /2     |
|  8 × 8         | dropout 0.5            |
|  8 × 8         | conv 3x3, 192          |
|  8 × 8         | conv 1x1, 192          |
|  8 × 8         | conv 1x1, 10           |
|  8 × 8         | avg pooling 8x8 /1     |
| 10 or 100      | softmax                |

<br /> 

## Overfeat

**Paper**: <br />
[OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks](http://arxiv.org/abs/1312.6229)
(P. Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus, Y. LeCun, 21 Dec 2013) <br />

**Code**: <br />
[Original code by P. Sermanet](https://github.com/sermanet/OverFeat) <br />

**Description**: <br />
Winner of the [ILSVRC 2013](http://image-net.org/challenges/LSVRC/2013/) at the category *"Classification with Localization"*. Details of the network architecture described in the paper (*fast* and *accurate* model) can be found below. A **ReLU** non-linearity is applied to the output of every convolutional and fully-connected layer. Note, the convolution and max pooling layers are similar to AlexNet, but with the following differences: no contrast normalization is used, pooling regions are non-overlapping, and the *accurate* model has larger feature maps (36x36 and 15x15 instead of 27x27 and 13x13),
thanks to a smaller convolution filter stride (2 instead of 4). A larger stride is beneficial for speed (see *fast* model), but will hurt accuracy.

| Input Size \\(\hspace{0.5cm}\\) | *Fast* Model |   ||   | Input Size \\(\hspace{0.5cm}\\) | *Accurate* Model |
|:-------------- |:----------------------------- |   ||   |:-------------- |:--------------------------------- |
| 231 × 231      | conv 11x11 /4, 96             |   ||   | 221 × 221      | conv 7x7 /2, 96                   |
| 56 × 56        | max pooling 2x2 /2            |   ||   | 110 × 110      | max pooling 3x3 /3                |
| 24 × 24        | conv 5x5 /1, 256              |   ||   | 36 × 36        | conv 7x7 /1, 256                  |
| 24 × 24        | max pooling 2x2 /2            |   ||   | 36 × 36        | max pooling 2x2 /2                |
| 12 × 12        | conv 3x3 /1, 512              |   ||   | 15 × 15        | conv 3x3 /1, 512                  |
| 12 × 12        | conv 3x3 /1, 1024             |   ||   | 15 × 15        | conv 3x3 /1, 512                  |
| 12 × 12        | conv 3x3 /1, 1024             |   ||   | 15 × 15        | conv 3x3 /1, 1024                 |
| 12 × 12        | max pooling 2x2 /2            |   ||   | 15 × 15        | conv 3x3 /1, 1024                 |
| 6 × 6 × 1024   | fc, 3072                      |   ||   | 15 × 15        | max pooling 3x3 /3                |
| 4096 × 1       | dropout 0.5                   |   ||   | 5 × 5 × 1024   | fc, 4096                          |
| 4096 × 1       | fc, 4096                      |   ||   | 4096 × 1       | dropout 0.5                       |
| 4096 × 1       | dropout 0.5                   |   ||   | 4096 × 1       | fc, 4096                          |
| 1000           | softmax                       |   ||   | 4096 × 1       | dropout 0.5                       |
|                |                               |   ||   | 1000           | softmax                           |

<br /> 

## SPPNet

**Paper**: <br />
[Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](http://arxiv.org/abs/1406.4729)
(K. He, X. Zhang, S. Ren and J. Sun; 18 Jun 2014) <br /> 

**Code**: <br /> 

**Description:**  <br />


## R-CNN 

**Paper:** <br />
[Rich feature hierarchies for accurate object detection and semantic segmentation](http://arxiv.org/abs/1311.2524) ((R. Girshick, J. Donahue, T. Darrell and J. Maliket; 11 Nov 2013) <br /> 

**Code:** <br /> 
[Original code by R. Girshick](https://github.com/rbgirshick/rcnn) <br /> 

**Description:** <br /> 
The R-CNN (= Regions with CNN-Features) method results in a better *"mean average precision"* (mAP) than Overfeat, but is still more time consuming.
It is applied to **AlexNet** and the 16 layer VGGNet (**VGG16**).


## Fast R-CNN

**Paper:** <br /> 
[Fast R-CNN](http://arxiv.org/abs/1504.08083) (R. Girshick; 30 Apr 2015) <br /> 

**Code:** <br />
[Original code by R. Girshick](https://github.com/rbgirshick/fast-rcnn) <br /> 

**Description:** <br />
The fast R-CNN method is a fusion of **R-CNN** and **SPPNet**. It results in an mAP better than Overfeat with similar computing time. Here, the fast R-CNN method uses the ConvNet architectures **AlexNet**, **VGG_CNN_M_1024** and **VGG16**. VGG_CNN_M_1024 is introduced in the paper [Return of the Devil in the Details: Delving Deep into Convolutional Nets](http://arxiv.org/pdf/1405.3531v4.pdf). Though, here FC7 consists of 1024 units, instead of 4096 units as described in the original paper.


## Faster R-CNN 

**Paper:** <br />
[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/abs/1506.01497) (S. Ren, K. He, R. Girshick and J. Sun; 4 Jun 2015) <br /> 

**Code:** <br />
[Original code by S. Ren](https://github.com/ShaoqingRen/faster_rcnn) <br /> 

**Description:** <br />
This is a convolutional network end-to-end implementation of fast R-CNN, i.e. even the region proposals are generated via the convolutional network. Thus, the selective search algorithm for region proposal generation is not needed any more. The faster R-CNN method is applied to the network architectures **ZFNet** and **VGG16**.


## Spatial Transformer Networks

**Paper:** <br />
[Spatial Transformer Networks](http://arxiv.org/abs/1506.02025) (M. Jaderberg, K. Simonyan, A. Zisserman and K. Kavukcuoglu; 5 Jun 2015) <br /> 

**Code:** <br />
["Torch" code](https://github.com/qassemoquab/stnbhwd) and ["Lasagne" code](http://lasagne.readthedocs.org/en/latest/modules/layers/special.html#lasagne.layers.TransformerLayer) (also see [this github repository](https://github.com/skaae/transformer_network)) <br /> 

**Description:** <br />


## Recurrent Spatial Transformer Networks

**Paper:** <br />
[Recurrent Spatial Transformer Networks](http://arxiv.org/abs/1509.05329) (S. Kaae Sønderby, C. Kaae Sønderby, L. Maaløe and O. Winther; 17 Sept 2015) <br /> 

**Code:** <br />
[Original code by S. Kaae Sønderby](https://github.com/skaae/recurrent-spatial-transformer-code) <br /> 

**Description:** <br />


## Fractional Max Pooling

**Paper**: <br />
[Fractional Max-Pooling](http://arxiv.org/abs/1412.6071) (B. Graham; 18 Dec 2014) <br />

**Code**: <br />
[See this discussion](https://github.com/torch/nn/issues/371) <br />

**Description**: <br />


## Further Readings

Additional paper regarding **R-CNN**, **Fast R-CNN** und **Faster R-CNN**: [R-CNN minus R](http://www.robots.ox.ac.uk/~vedaldi/assets/pubs/lenc15rcnn.pdf) (K. Lenc
and A. Vedaldi; University of Oxford, 23 Jun 2015)

Github blog on [Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/convolutional-networks): Some prominent convolutional networks are listet at the bottom of this page.


