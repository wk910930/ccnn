#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2017 CUHK
# Licensed under The MIT License [see LICENSE for details]
# Written by Kun Wang
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import argparse
import pprint
import os
import sys
import cPickle
import _init_paths
from datasets.factory import get_imdb

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--aboxes', dest='aboxes',
                        help='aboxes file(s) path',
                        nargs='+', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print 'Called with args:'
    print args

    imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(True)

    base_dir = os.path.split(args.aboxes[0])[0]
    output_dir = os.path.join(base_dir, 'collect_and_eval')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    for abox_name in args.aboxes:
        with open(abox_name) as f:
            tmp = cPickle.load(f)
            for im_ind, index in enumerate(imdb.image_index):
                for cls_ind, cls in enumerate(imdb.classes):
                   dets = tmp[cls_ind][im_ind]
                   if dets == []:
                       continue
                   all_boxes[cls_ind][im_ind] = dets

    print 'Evaluating...'
    imdb.evaluate_detections(all_boxes, output_dir)
