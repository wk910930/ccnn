#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2016 CUHK
# Written by Kun Wang
# --------------------------------------------------------

"""
Demo script showing detections in given dataset.
"""

import os
import sys
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Visualize ilsvrc txt result')
    parser.add_argument('--txt', dest='txt_file',
                        help='txt file generated for final submition',
                        default=None, type=str)
    parser.add_argument('--names', dest='file_names',
                        help='specify images to visualize if given',
                        default=None, type=str, nargs='*')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

CLASSES = ('__background__',  # always index 0
           'accordion', 'airplane', 'ant', 'antelope', 'apple',
           'armadillo', 'artichoke', 'axe', 'baby bed', 'backpack',
           'bagel', 'balance beam', 'banana', 'band aid', 'banjo',
           'baseball', 'basketball', 'bathing cap', 'beaker', 'bear',
           'bee', 'bell pepper', 'bench', 'bicycle', 'binder',
           'bird', 'bookshelf', 'bow', 'bow tie', 'bowl',
           'brassiere', 'burrito', 'bus', 'butterfly', 'camel',
           'can opener', 'car', 'cart', 'cattle', 'cello',
           'centipede', 'chain saw', 'chair', 'chime', 'cocktail shaker',
           'coffee maker', 'computer keyboard', 'computer mouse', 'corkscrew', 'cream',
           'croquet ball', 'crutch', 'cucumber', 'cup or mug', 'diaper',
           'digital clock', 'dishwasher', 'dog', 'domestic cat', 'dragonfly',
           'drum', 'dumbbell', 'electric fan', 'elephant', 'face powder',
           'fig', 'filing cabinet', 'flower pot', 'flute', 'fox',
           'french horn', 'frog', 'frying pan', 'giant panda', 'goldfish',
           'golf ball', 'golfcart', 'guacamole', 'guitar', 'hair dryer',
           'hair spray', 'hamburger', 'hammer', 'hamster', 'harmonica',
           'harp', 'hat with a wide brim', 'head cabbage', 'helmet', 'hippopotamus',
           'horizontal bar', 'horse', 'hotdog', 'iPod', 'isopod',
           'jellyfish', 'koala bear', 'ladle', 'ladybug', 'lamp',
           'laptop', 'lemon', 'lion', 'lipstick', 'lizard',
           'lobster', 'maillot', 'maraca', 'microphone', 'microwave',
           'milk can', 'miniskirt', 'monkey', 'motorcycle', 'mushroom',
           'nail', 'neck brace', 'oboe', 'orange', 'otter',
           'pencil box', 'pencil sharpener', 'perfume', 'person', 'piano',
           'pineapple', 'ping-pong ball', 'pitcher', 'pizza', 'plastic bag',
           'plate rack', 'pomegranate', 'popsicle', 'porcupine', 'power drill',
           'pretzel', 'printer', 'puck', 'punching bag', 'purse',
           'rabbit', 'racket', 'ray', 'red panda', 'refrigerator',
           'remote control', 'rubber eraser', 'rugby ball', 'ruler', 'salt or pepper shaker',
           'saxophone', 'scorpion', 'screwdriver', 'seal', 'sheep',
           'ski', 'skunk', 'snail', 'snake', 'snowmobile',
           'snowplow', 'soap dispenser', 'soccer ball', 'sofa', 'spatula',
           'squirrel', 'starfish', 'stethoscope', 'stove', 'strainer',
           'strawberry', 'stretcher', 'sunglasses', 'swimming trunks', 'swine',
           'syringe', 'table', 'tape player', 'tennis ball', 'tick',
           'tie', 'tiger', 'toaster', 'traffic light', 'train',
           'trombone', 'trumpet', 'turtle', 'tv or monitor', 'unicycle',
           'vacuum', 'violin', 'volleyball', 'waffle iron', 'washer',
           'water bottle', 'watercraft', 'whale', 'wine bottle', 'zebr')

def class_to_ind():
    classes = ('n02672831', 'n02691156', 'n02219486', 'n02419796', 'n07739125',
                     'n02454379', 'n07718747', 'n02764044', 'n02766320', 'n02769748',
                     'n07693725', 'n02777292', 'n07753592', 'n02786058', 'n02787622',
                     'n02799071', 'n02802426', 'n02807133', 'n02815834', 'n02131653',
                     'n02206856', 'n07720875', 'n02828884', 'n02834778', 'n02840245',
                     'n01503061', 'n02870880', 'n02883205', 'n02879718', 'n02880940',
                     'n02892767', 'n07880968', 'n02924116', 'n02274259', 'n02437136',
                     'n02951585', 'n02958343', 'n02970849', 'n02402425', 'n02992211',
                     'n01784675', 'n03000684', 'n03001627', 'n03017168', 'n03062245',
                     'n03063338', 'n03085013', 'n03793489', 'n03109150', 'n03128519',
                     'n03134739', 'n03141823', 'n07718472', 'n03797390', 'n03188531',
                     'n03196217', 'n03207941', 'n02084071', 'n02121808', 'n02268443',
                     'n03249569', 'n03255030', 'n03271574', 'n02503517', 'n03314780',
                     'n07753113', 'n03337140', 'n03991062', 'n03372029', 'n02118333',
                     'n03394916', 'n01639765', 'n03400231', 'n02510455', 'n01443537',
                     'n03445777', 'n03445924', 'n07583066', 'n03467517', 'n03483316',
                     'n03476991', 'n07697100', 'n03481172', 'n02342885', 'n03494278',
                     'n03495258', 'n03124170', 'n07714571', 'n03513137', 'n02398521',
                     'n03535780', 'n02374451', 'n07697537', 'n03584254', 'n01990800',
                     'n01910747', 'n01882714', 'n03633091', 'n02165456', 'n03636649',
                     'n03642806', 'n07749582', 'n02129165', 'n03676483', 'n01674464',
                     'n01982650', 'n03710721', 'n03720891', 'n03759954', 'n03761084',
                     'n03764736', 'n03770439', 'n02484322', 'n03790512', 'n07734744',
                     'n03804744', 'n03814639', 'n03838899', 'n07747607', 'n02444819',
                     'n03908618', 'n03908714', 'n03916031', 'n00007846', 'n03928116',
                     'n07753275', 'n03942813', 'n03950228', 'n07873807', 'n03958227',
                     'n03961711', 'n07768694', 'n07615774', 'n02346627', 'n03995372',
                     'n07695742', 'n04004767', 'n04019541', 'n04023962', 'n04026417',
                     'n02324045', 'n04039381', 'n01495701', 'n02509815', 'n04070727',
                     'n04074963', 'n04116512', 'n04118538', 'n04118776', 'n04131690',
                     'n04141076', 'n01770393', 'n04154565', 'n02076196', 'n02411705',
                     'n04228054', 'n02445715', 'n01944390', 'n01726692', 'n04252077',
                     'n04252225', 'n04254120', 'n04254680', 'n04256520', 'n04270147',
                     'n02355227', 'n02317335', 'n04317175', 'n04330267', 'n04332243',
                     'n07745940', 'n04336792', 'n04356056', 'n04371430', 'n02395003',
                     'n04376876', 'n04379243', 'n04392985', 'n04409515', 'n01776313',
                     'n04591157', 'n02129604', 'n04442312', 'n06874185', 'n04468005',
                     'n04487394', 'n03110669', 'n01662784', 'n03211117', 'n04509417',
                     'n04517823', 'n04536866', 'n04540053', 'n04542943', 'n04554684',
                     'n04557648', 'n04530566', 'n02062744', 'n04591713', 'n02391049')
    cls_ind = dict(zip(classes, xrange(200)))
    return cls_ind

def load_det_list(det_list):
    file_names = []
    with open(det_list) as f:
        for line in f:
            file_names.append(line.rstrip('\n'))
    return file_names

def load_dets(txt_file):
    """
    dets: [cls_id, score, x1, y1, x2, y2]
    """
    dets = np.loadtxt(txt_file, delimiter=' ')
    return dets

def load_gt(ann_file):
    tree = ET.parse(ann_file)
    objs = tree.findall('object')
    gt_dets = np.ones((len(objs), 5))
    label_idx = []
    cls_ind = class_to_ind()
    for ix, obj in enumerate(objs):
        obj_name = obj.find('name').text
        x1 = obj.find('bndbox').find('xmin').text
        y1 =  obj.find('bndbox').find('ymin').text
        x2 = obj.find('bndbox').find('xmax').text
        y2 = obj.find('bndbox').find('ymax').text
        gt_dets[ix, 0:4] = np.array([x1, y1, x2, y2])
        label_idx.append(cls_ind[obj_name] + 1)
    return gt_dets, label_idx

def vis_detections(im, class_name, dets, bbox_color='blue', thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor=bbox_color, linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(CLASSES[class_name[i]], score),
                bbox=dict(facecolor=bbox_color, alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('score >= {:.1f}').format(thresh), fontsize=14)
    plt.axis('off')
    plt.tight_layout()

if __name__ == '__main__':
    args = parse_args()

    print 'Called with args:'
    print args

    det_list = '/home/kwang/Documents/ImageNet/det_lists/13val2.txt'
    im_dir = '/home/kwang/Documents/ImageNet/ILSVRC2014/ILSVRC2013_DET_val'
    ann_dir = '/home/kwang/Documents/ImageNet/ILSVRC2014/ILSVRC2013_DET_bbox_val'
    txt_file = args.txt_file
    # load txt file
    dets = load_dets(txt_file)
    # load image names
    file_names = load_det_list(det_list)
    # specify view order
    view_orders = []
    if args.file_names:
        for j in args.file_names:
            view_orders.append(file_names.index(j))
    else:
        view_orders = np.random.permutation(len(file_names))

    for im_ind in view_orders:
        file_name = file_names[im_ind]
        print '{} {}'.format(im_ind + 1, file_name)

        im_file = os.path.join(im_dir, file_name) + '.JPEG'
        im = cv2.imread(im_file)
        tmp_dets = dets[dets[:, 0] == im_ind+1, 2:]
        tmp_cls_ids = dets[dets[:, 0] == im_ind+1, 1].astype(int)
        tmp_dets = tmp_dets[:, (1, 2, 3, 4, 0)]

        # load ground truth
        gt_obj, label_idx = load_gt(os.path.join(ann_dir, file_name) + '.xml')

        vis_detections(im, tmp_cls_ids, tmp_dets, bbox_color='blue', thresh=0.1)
        vis_detections(im, label_idx, gt_obj, bbox_color='red', thresh=0.1)
        plt.draw()
        plt.show()
