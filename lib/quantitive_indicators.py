#!/usr/bin/env python3
import numpy as np
from lib.util import cal_mask_ratio


def calculate_q(srcRes, tarRes, detRes,
                threshold_interval, threshold_step_size, return_dict=False):
    """
    calculate Q averaged of all threshold
    :param srcRes: source encoder phis ?
    :param tarRes: target encoder phit ?
    :param detRes: deepfake detector phid ?
    :param threshold_interval: 0.6 -> 1 by default
    :param threshold_step_size: 0.05 by default
    :return: Q averaged of all threshold
    """

    Q_result = dict()
    threshold, max_threshold = threshold_interval

    while threshold <= max_threshold:
        ratio = cal_mask_ratio(srcRes, tarRes, detRes, threshold)
        Q_result[str(threshold)] = ratio
        threshold += threshold_step_size
    if return_dict:
        return Q_result
    mean_Q = np.mean(list(Q_result.values()))
    return mean_Q


def stability(raw_score, c23_score, c40_score):
    """
    calculate delta
    :param raw_score: phi_raw
    :param c23_score: phi_c23
    :param c40_score: phi_c40
    :return: delta
    """
    result_list = list()
    for key, value in raw_score.items():
        raw = np.array(value.reshape(-1))
        c23 = np.array(c23_score[key.replace("raw", "c23")]).reshape(-1)
        c40 = np.array(c40_score[key.replace("raw", "c40")]).reshape(-1)

        raw_c23 = (c23 * raw).sum()
        raw_c40 = (c40 * raw).sum()
        result_list.append((raw_c23 + raw_c40) / 2)
    return sum(result_list) / len(result_list)

# vim: ts=4 sw=4 sts=4 expandtab
