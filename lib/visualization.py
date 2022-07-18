import argparse
import cv2
import json
import os
import yaml

import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.pyplot import subplots_adjust

import megengine as mge
import megengine.functional as F

from lib.util import generate_mask

matplotlib.use('agg')


def draw_result(final_result, save_path):
    """
    visualize Q of different threshold for pair and unpair
    :param final_result: dict of Q, e.g. {str{threshold}: [Qpair, Qunpair]}
    :param save_path: save path
    :return: None
    """
    for deepfake_method, result in final_result.items():
        data = list()
        thresholds = [float(i) for i in list(result.keys())]

        data = [value for value in result.values()]

        data = np.array(data)
        thresholds = np.array(thresholds)

        sns.set(rc={'figure.figsize': (5, 4)})
        sns.set_style('whitegrid')
        wide_df = pd.DataFrame(
            data, thresholds,
            [
                "Classifier with Pair Training Set",
                "Classifier with Unpair Training Set"
            ]
        )

        plt.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
        sns.set_context("notebook", rc={"lines.linewidth": 2.5})
        fig = sns.lineplot(markers=[".", '.'], data=wide_df)
        fig = fig.get_figure()
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(f"{save_path}/{deepfake_method}_pair_comparison.png")

        plt.close(fig)


def do_cat_imgs(cat_imgs, path):
    """
    visualize img, phis, phit, phid, intersection
    :param feature: cat_imgs, e.g. [img, phis, phit, phid, intersection]
    :param path: save path
    :return: None
    """
    img_num = len(cat_imgs)
    mycolor = ['#F5F5F5', '#9FDEDE', 'orange', 'maroon']
    self_cmap = colors.LinearSegmentedColormap.from_list('my_list', mycolor)
    fig, axes = plt.subplots(1, img_num, figsize=(2*img_num, 2))

    for col in range(img_num):
        axes[col].axis("off")
        if col != 0:
            if col == img_num - 1:
                axes[col].imshow(cat_imgs[col], cmap=self_cmap)
            else:
                axes[col].imshow(
                    cat_imgs[col], cmap='bwr', vmin=-0.2, vmax=0.2
                )
        else:
            axes[col].imshow(cat_imgs[col])

    subplots_adjust(
        left=0.02, bottom=0.02, right=0.98,
        top=0.98, wspace=0.05, hspace=0.05
    )
    plt.savefig(path)
    plt.cla()
    plt.clf()


def visualize_phi(
        source_re, target_re, det_re, imgs, threshold, save_path
):
    """
    visualize_phi
    :param source_re: phis
    :param target_re: phit
    :param det_re: phid
    :param imgs: input imgs
    :param threshold: the precentage of grids to mask
    :param save_path: save path of the visualization
    :return: None
    """
    for index, (img_info, source_shapley) in enumerate(source_re.items()):
        # get score_value
        source_shapley = mge.Tensor(np.array(source_re[img_info]))
        target_shapley = mge.Tensor(np.array(target_re[img_info]))
        det_shapley = mge.Tensor(np.array(det_re[img_info]))
        # generate mask
        result_union = F.maximum(source_shapley, target_shapley)
        mask_union, _ = generate_mask(result_union, threshold)
        mask_det, result = generate_mask(det_shapley, threshold)

        mask_union = np.array(mask_union)
        mask_det = np.array(mask_det)
        intersection = mask_union + (mask_det * 2)
        input_img = cv2.cvtColor(imgs[index].numpy().transpose(1, 2, 0).astype('uint8'), cv2.COLOR_BGR2RGB)
        cat_imgs = [
            input_img, source_shapley, target_shapley, result, intersection
        ]
        os.makedirs(save_path, exist_ok=True)
        do_cat_imgs(cat_imgs, f"{save_path}/{img_info}_shap_vis.png")

# vim: ts=4 sw=4 sts=4 expandtab
