# Learning Chained Deep Features and Classifiers for Cascade in Object Detection

by Wanli Ouyang, Kun Wang, Xin Zhu and Xiaogang Wang.

## Introduction

This paper presents chained cascade network (CC-Net). In this CC-Net, the cascaded classifier at a stage is aided by the
classification scores in previous stages. Feature chaining is further proposed so that the feature learning for the current
cascade stage uses the features in previous stages as the prior information. The chained ConvNet features and classifiers of
multiple stages are jointly learned in an end-to-end network. In this way, features and classifiers at latter stages handle
more difficult samples with the help of features and classifiers in previous stages. It yields consistent boost in detection
performance on benchmarks like PASCAL VOC 2007 and ImageNet. Combined with better region proposal, CC-Net leads to state-of-the-art result of 81.1% mAP on PASCAL VOC2007. For more details, please refer to our
[arXiv paper](http://arxiv.org/abs/1702.07054).

## Method

<p align="center">
<img src="figure1.png" alt="Motivation">
</p>

## Cascade Loss

```
layer {
  name: "loss_cls_cas_128"
  type: "SoftmaxWithCascadeLoss"
  bottom: "cls_score_cas_128"
  bottom: "labels"
  bottom: "bp_map_cas_128"
  top: "loss_cls_cas_128"
  top: "bp_map_cas_64"
  loss_weight: 1
  loss_weight: 0
  loss_param {
    hard_mining: true
    sampling: true
    cascade: true
    bp_size: 64
    cascade_type: 1
    threshold: 0.9
    batch_size: 60
    gt_batch_size: 2
    ims_per_batch: 1
    gt_per_batch: 2
    fg_fraction: 0.25
  }
}
```

### Citation

If you find the code or the models useful, please cite this paper:
```
@article{ouyang2017learning,
  title={Learning Chained Deep Features and Classifiers for Cascade in Object Detection},
  author={Ouyang, Wanli and Wang, Kun and Zhu, Xin and Wang, Xiaogang},
  journal={arXiv preprint arXiv:1702.07054},
  year={2017}
}
```
