#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2016 CUHK
# Written by Wang Kun
# --------------------------------------------------------

"""Generate txt file as the input to craft::frcnn_train_data_layer"""

import cPickle
import PIL
import _init_paths
from datasets.ilsvrc import ilsvrc
from roi_data_layer.roidb import add_bbox_regression_targets

def prepare_gt_roidb(imdb, gt_roidb):
    """
    """
    sizes = [PIL.Image.open(imdb.image_path_at(i)).size
             for i in xrange(imdb.num_images)]
    bands = [PIL.Image.open(imdb.image_path_at(i)).getbands()
             for i in xrange(imdb.num_images)]
    for i in xrange(len(imdb.image_index)):
        gt_roidb[i]['image'] = imdb.image_path_at(i)
        gt_roidb[i]['width'] = sizes[i][0]
        gt_roidb[i]['height'] = sizes[i][1]
        gt_roidb[i]['channel'] = len(bands[i])

def main():
    image_set = 'val1'
    year = '2013'
    proposal_method = 'slide'

    imdb = ilsvrc(image_set, year)
    gt_roidb = imdb.gt_roidb()
    prepare_gt_roidb(imdb, gt_roidb)

    # filter roidb
    print 'filtering out images with no gt...'
    remove_indices = []
    for i in xrange(len(gt_roidb)):
        if gt_roidb[i]['boxes'].shape[0] == 0:
            remove_indices.append(i)
    gt_roidb = [i for j, i in enumerate(gt_roidb) if j not in remove_indices]
    print '{} images are filtered'.format(len(remove_indices))

    with open('rois_ilsvrc_' + image_set + '_' + year + '.txt', 'w') as f:
        for image_index in xrange(len(gt_roidb)):
            if image_index % 1000 == 0:
                print '{}/{}'.format(image_index, len(gt_roidb))

            # load next item
            item = gt_roidb[image_index]
            # image_index img_path channels height width
            f.write('# {}\n{}\n{}\n{}\n{}\n'.format(
                image_index, item['image'], item['channel'], item['height'], item['width']))
            # flipped
            f.write('{}\n'.format(0))
            # num_windows
            num_windows = item['boxes'].shape[0]
            f.write('{}\n'.format(num_windows))
            for k in xrange(num_windows):
                # class_index
                class_index = item['gt_classes'][k]
                f.write('{} '.format(class_index))
                # x1 y1 x2 y2
                x1 = item['boxes'][k, 0]
                y1 = item['boxes'][k, 1]
                x2 = item['boxes'][k, 2]
                y2 = item['boxes'][k, 3]
                f.write('{} {} {} {}\n'.format(x1, y1, x2, y2))

if __name__ == '__main__':
    main()
