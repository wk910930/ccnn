#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import _init_paths
from fast_rcnn.test import test_net, _get_blobs
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv, bbox_transform, bbox_voting
from utils.cython_bbox import bbox_overlaps
from datasets.factory import get_imdb
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import caffe
import argparse
import pprint
import time, os, sys
import numpy as np
import cPickle

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--aboxes', dest='aboxes',
                        help='aboxes file path',
                        default=None, type=str)
    parser.add_argument('--nms', dest='nms',
                        help='NMS threshold',
                        default=None, type=float)
    parser.add_argument('--voting', dest='voting',
                        help='bounding box voting',
                        default=None, type=int)
    parser.add_argument('--voting_thresh', dest='voting_thresh',
                        help='bounding box voting threshold',
                        default=None, type=float)
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.nms is not None:
        cfg.TEST.NMS = args.nms
    if args.voting is not None:
        cfg.TEST.BBOX_VOTE = bool(args.voting)
    if args.voting_thresh is not None:
        cfg.TEST.BBOX_VOTE_THRESH = args.voting_thresh
    max_per_image = args.max_per_image

    print('Using config:')
    pprint.pprint(cfg)

    imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)

    base_dir = os.path.split(args.aboxes)[0]
    output_dir = os.path.join(base_dir, 'nms_'+str(cfg.TEST.NMS)+'_voting_'+str(cfg.TEST.BBOX_VOTE)+'_voting_thresh_'+str(cfg.TEST.BBOX_VOTE_THRESH))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.aboxes) as f:
        all_boxes = cPickle.load(f)

    num_images = len(imdb.image_index)

    for i in xrange(num_images):
        print 'Image', i+1, 'of', num_images
        sys.stdout.flush()
        # skip j = 0, because it's the background class
        for j in xrange(1, imdb.num_classes):
            cls_dets = all_boxes[j][i]
            keep = nms(cls_dets, cfg.TEST.NMS)
            if cfg.TEST.BBOX_VOTE:
                cls_dets_after_nms = cls_dets[keep, :]
                cls_dets = bbox_voting(cls_dets_after_nms, cls_dets, threshold=cfg.TEST.BBOX_VOTE_THRESH)
            else:
                cls_dets = cls_dets[keep]
            all_boxes[j][i] = cls_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

    print 'Evaluating...'
    imdb.evaluate_detections(all_boxes, output_dir)
