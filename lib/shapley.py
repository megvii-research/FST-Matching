import numpy as np
from itertools import combinations, chain
from keras.utils import to_categorical
import cv2
import os
from tqdm import tqdm
import megengine.functional as F
import megengine as mge


def turn_list(s):
    if type(s) == list:
        return s
    elif type(s) == int:
        return [s]


def get_sampleshapley(positions_dict):
    """
    Convert postion dict to key and val
    :param positions_dict: {(0, 0): [0,1, ..., 0], (0, 1): [0,0, ..., 0]}
    :return: key_to_idx: {(0, 0): [0,1, ..., 99], (0, 1): [100, ..., 199]}, positions: 200 x 256
    """
    keys, values = positions_dict.keys(), positions_dict.values()  # return a list of all values
    values = [np.array(value) for value in values]
    positions = np.concatenate(values, axis=0)  # [2*m, a_len]
    key_to_idx = {}
    count = 0
    for i, key in enumerate(keys):
        key_to_idx[key] = list(range(count, count + len(values[i])))
        count += len(values[i])
    return key_to_idx, positions


def get_masks_sampleshapley(clist, cIdx, m=100):
    """
    Sample patch to approximate shapley value
    :param clist: patch id list, e.g. [0, 1, 2, ..., 255]
    :param cIdx: index of the patch to calculate the Shapley value, e.g. 10
    :param m: sample times.
    :return: positions_dict, which denotes the index of the sampled patch
    """
    d = len(clist)
    positions_dict = {(i, fill): [] for i in range(1) for fill in [0, 1]}
    # sample m times
    for cnt in range(m):
        perm = np.random.permutation(d)
        preO = []
        for idx in perm:
            if idx != cIdx:
                preO.append(turn_list(clist[idx]))
            else:
                break
        preO_list = list(chain.from_iterable(preO))
        pos_excluded = np.sum(to_categorical(preO_list, num_classes=d), axis=0)
        pos_included = pos_excluded + np.sum(to_categorical(turn_list(clist[cIdx]), num_classes=d), axis=0)
        positions_dict[(0, 0)].append(pos_excluded)
        positions_dict[(0, 1)].append(pos_included)
    return positions_dict


def explain_shapley(predict, x, key_to_idx, label):
    """
    Forward sampled patches to calculate the  Shapley value
    :param predict: DNN model
    :param x: masked images
    :param key_to_idx: position dict
    :param label: the ground truth label
    :return: phis (the Shapley value)
    """
    labels = [label for i in range(x.shape[0])]
    logits = predict(x).detach().numpy()
    # one-hot vector to denote the refer class of x
    discrete_probs = np.eye(len(logits[0]))[labels]
    vals = np.sum(discrete_probs * logits, axis=1)
    # key_to_idx[key]: list of indices in original position
    key_to_val = {key: np.array([vals[idx] for idx in key_to_idx[key]]) for key in key_to_idx}
    # Compute importance scores
    phis = (key_to_val[(0, 1)] - key_to_val[(0, 0)]).mean()
    return phis


# normalize the shapley value vector to unit vector
def normalize(result):
    norm_factor = F.sqrt(F.sum(result ** 2))
    result = result / norm_factor
    return result


def position_2_mask(positions, feature, grid_scale):
    """
    Convert position dict to masked images
    :param position: position dict
    :param feature: input images (C * H * W)
    :param grid_scale: num of patch per row/column, e.g. 16
    :return: mask (sample_times * C * H * W)
    """
    C, H, W = feature.shape
    img_size = feature.shape[-1]
    sample_times = positions.shape[0]
    grid_size = img_size // grid_scale
    mask = F.repeat(F.expand_dims(F.ones_like(feature), axis=0), repeats=sample_times, axis=0)
    mask = mask.reshape(sample_times, C, grid_scale, grid_size, grid_scale, grid_size)
    mask = mask.transpose(0, 1, 2, 4, 3, 5)
    mask = mask.reshape(sample_times, C, grid_scale*grid_scale, grid_size, grid_size)
    mask = mask * mge.Tensor(positions).to(feature.device).reshape(sample_times, 1, grid_scale*grid_scale, 1, 1)
    mask = mask.reshape(sample_times, C, grid_scale, grid_scale, grid_size, grid_size)
    mask = mask.transpose(0, 1, 2, 4, 3, 5)
    mask = mask.reshape(sample_times, C, H, W)
    return mask


def shapley_function_visual(feature, predict, label, img_info, sample_times=100, grid_scale=16):
    """
    Get the Shapley value for a batch of images
    :param feature: input images (B * C * H * W)
    :param predict: DNN model
    :param label: the ground truth label (B * 1)
    :param img_info: the list of path of imgs (B * 1)
    :param sample times: sample times.
    :param grid_scale: num of patch per row/column, e.g. 16
    :return: result_lis (the Shapley value of all patches of all images)
    """

    B = len(label)
    result_lis = dict()
    for i in tqdm(range(B)):
        result = F.zeros((grid_scale, grid_scale))
        for id in tqdm(range(grid_scale * grid_scale)):
            positions_dict = get_masks_sampleshapley(clist=[i for i in range(grid_scale**2)], cIdx=id, m=sample_times)
            key_to_idx, positions = get_sampleshapley(positions_dict)
            mask = position_2_mask(positions, feature[i], grid_scale)
            inputs = F.expand_dims(feature[i], axis=0) * mask
            shaps = explain_shapley(predict, inputs, key_to_idx, label[i])
            result[id // grid_scale, id % grid_scale] = shaps
        result = normalize(result)
        result_lis[img_info[i]] = result
    return result_lis


if __name__ == '__main__':
    feature = mge.tensor(np.zeros((2, 3, 224, 224)), dtype="float32")
    predict = None
    label = mge.Tensor([1, 1])
    shapley_lis = shapley_function_visual(feature, predict, label)
