#!/usr/bin/env python3
import json
import cv2
import numpy as np
import os
from PIL import Image
import megengine as mge

from dataset.augmentor import align_5p


def get_align5p(img, ld):
    img, = align_5p(
        img, ld=ld, face_width=80,
        canvas_size=224, scale=0.9
    )
    return img


def load_imgs(img_path, deepfake, img_type):
    """
    load and preprocess the images
    :param image_path: ./imgs
    :param deepfake: DF/F2F/FS/FSh/NT
    :param image_type: raw/c23/c40
    :return: imgs, img_info, det_lable, source_label, target_label
    """
    # read the landmark
    with open(f"{img_path}/ldm.json", 'r') as f:
        ldm_dict = json.load(f)
    # read the images
    img_full_path = f"{img_path}/manipulated_sequences/{deepfake}/{img_type}/frames"
    imgs_path_list = os.listdir(img_full_path)
    imgNum = len(imgs_path_list)
    img_info = list()
    data = np.empty((imgNum, 3, 224, 224), dtype="float32")
    det_label = np.ones((imgNum,), dtype="int32")
    source_label = np.empty((imgNum, ), dtype="int32")
    target_label = np.empty((imgNum, ), dtype="int32")
    for i, img_path in enumerate(imgs_path_list):
        img_path = os.path.join(img_full_path, img_path, "frame_0.png")
        img = cv2.imread(img_path)
        img = np.asarray(img, dtype="float32")
        info_key = img_path[:-4]
        ld = np.array(ldm_dict[info_key]['landmark'])
        source_label[i] = ldm_dict[info_key]['source_label']
        target_label[i] = ldm_dict[info_key]['target_label']
        # crop and align face
        img = get_align5p(img, ld).astype('uint8')
        # 224x224x3 - > 3x224x224
        img = img.transpose(2, 0, 1)
        data[i, :, :, :] = img
        img_info.append(f"{deepfake}_{img_type}_{source_label[i]}_{target_label[i]}")
    data = mge.Tensor(data)
    return data, img_info, det_label, source_label, target_label
    
# vim: ts=4 sw=4 sts=4 expandtab
