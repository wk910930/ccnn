#!/bin/bash

./tools/test_net.py --gpu 0 \
     --def /home/kwang/model_zoo/detections/bn_liuyu/deploy.prototxt \
     --net /home/kwang/model_zoo/detections/bn_liuyu/models/hkbn_4d_fast_rcnn_iter_120000.caffemodel \
     --imdb ilsvrc_2013_val2 \
     --comp \
     --thresh 0.05 \
     --cfg ./experiments/cfgs/frcnn.yml \
     --bbox_mean /home/kwang/proposals/ilsvrc/liuyu/bbox_means.pkl \
     --bbox_std /home/kwang/proposals/ilsvrc/liuyu/bbox_stds.pkl
