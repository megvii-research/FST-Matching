#!/usr/bin/env python3

import yaml
import numpy as np

import megengine as mge
import megengine.functional as F


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.load(f)
    return config


def generate_mask(result, threshold=0.8):
    """
    generate mask
    :param result: phi (16x16)
    :param threshold: the precentage of grids to mask
    :return: mask (16x16), result
    """
    size = result.shape
    grid_num = size[0] * size[1]
    result_before_sort = result.reshape(grid_num)
    result_before_sort = mge.functional.nn.relu(result_before_sort)
    result_after_sort, idx = F.sort(result_before_sort)
    threshold = result_before_sort[idx[int(grid_num * threshold)]]
    mask = (result > threshold).reshape(size)
    return mask, result


def cal_mask_ratio(
        source_re, target_re, det_re, threshold
):
    """
    calculate Q for a certain threshold of all images
    :param source_re: phis
    :param target_re: phit
    :param det_re: phid
    :param threshold: the precentage of grids to mask
    :return: Q averaged of all images
    """
    ratios = list()
    for img_info, source_shapley in source_re.items():
        # get score_value
        source_shapley = mge.Tensor(np.array(source_re[img_info]))
        target_shapley = mge.Tensor(np.array(target_re[img_info]))
        det_shapley = mge.Tensor(np.array(det_re[img_info]))
        # generate mask
        result_union = F.maximum(source_shapley, target_shapley)
        mask_union, _ = generate_mask(result_union, threshold)
        mask_det, result = generate_mask(det_shapley, threshold)

        attention_weight = np.array(result)
        mask_union = np.array(mask_union)
        ratio = ((1-mask_union) * attention_weight).mean() - (mask_union * attention_weight).mean()
        ratios.append(ratio)
    mean_ratios = np.mean(ratios)
    return mean_ratios

# vim: ts=4 sw=4 sts=4 expandtab
