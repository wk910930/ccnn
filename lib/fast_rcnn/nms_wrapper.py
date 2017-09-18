# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
from fast_rcnn.config import cfg
from nms.gpu_nms import gpu_nms
from nms.cpu_nms import cpu_nms, cpu_soft_nms

def soft_nms(dets, sigma=0.5, Nt=0.3, threshold=0.001, method=1):
    """
    Doing Soft-NMS.
    Check https://github.com/bharatsingh430/soft-nms for details.

    Parameters
    ----------
        dets:
        sigma:
        Nt:
        threshold:
        method: 1 for linear, 2 for gaussian and others for original NMS

    Returns
    -------
    """

    keep = cpu_soft_nms(np.ascontiguousarray(dets, dtype=np.float32),
                        np.float32(sigma), np.float32(Nt),
                        np.float32(threshold),
                        np.uint8(method))
    return keep

def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    if cfg.USE_GPU_NMS and not force_cpu:
        return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
    else:
        return cpu_nms(dets, thresh)
