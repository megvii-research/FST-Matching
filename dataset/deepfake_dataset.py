#!/usr/bin/env python3
import os
import cv2
import json
import numpy as np
from typing import Dict, List, Tuple
from megengine.data.dataset import VisionDataset

from dataset import augmentor


class DeepfakeDataset(VisionDataset):
    r"""DeepfakeDataset Dataset.

    The folder is expected to be organized as followed: root/cls/xxx.img_ext

    Labels are indices of sorted classes in the root directory.

    Args:
        root: root directory of an image folder.
        landmark_json: path to landmark json.
        class_name: if is True, return class name instead of class index.
    """

    def __init__(self, root: str, landmark_json: str, model_type: str, mode: str, class_name: bool = False,):
        super().__init__(root, order=("image", "image_category"))

        self.root = root
        self.class_name = class_name
        self.rng = np.random
        assert mode in ['train', 'test']
        self.do_train = True if model_type == 'FSTMatching' and mode=='train' else False
        self.do_warp = True if model_type == 'FSTMatching' and mode=='train' else False
        self.landmark_path = landmark_json
        self.info_meta_dict = self.load_landmark_json(self.landmark_path)
        self.class_dict = self.collect_class()
        self.samples = self.collect_samples()

    def load_landmark_json(self, landmark_json) -> Dict:
        with open(landmark_json, 'r') as f:
            landmark_dict = json.load(f)
        return landmark_dict

    def collect_samples(self) -> List:
        samples = []
        directory = os.path.expanduser(self.root)
        for key in sorted(self.class_dict.keys()):
            d = os.path.join(directory, key)
            if not os.path.isdir(d):
                continue
            for r, _, filename in sorted(os.walk(d, followlinks=True)):
                for name in sorted(filename):
                    path = os.path.join(r, name)
                    info_key = path[:-4]
                    info_meta = self.info_meta_dict[info_key]
                    landmark = info_meta['landmark']
                    source_label = int(info_meta['source_label'])
                    target_label = int(info_meta['target_label'])
                    class_label = int(info_meta['det_label'])
                    samples.append(
                        (path, {'Labels': class_label, 'SrcLabels': source_label,
                                'TarLabels': target_label, 'landmark': landmark})
                    )

        return samples

    def collect_class(self) -> Dict:
        classes = [d.name for d in os.scandir(self.root) if d.is_dir()]
        classes.sort()
        return {classes[i]: np.int32(i) for i in range(len(classes))}

    def prepare_input(self, img, ld, label, rng, do_train, do_warp):

        def get_align5p(img, ld, rng):
            img = augmentor.align_5p(
                img, ld=ld,
                face_width=80, canvas_size=224,
                scale=(rng.randn()*0.1+0.9 if do_train else 1),
                translation=([augmentor.rand_range(rng, -25, 25), augmentor.rand_range(rng, -25, 25)] if do_train else [0, 0]),
                rotation=(30*augmentor.rand_range(rng, -1, 1)**3 if do_train else 0),
                sa=(augmentor.rand_range(rng, .97, 1.03) if do_train and do_warp and rng.rand() > 0.8 else 1),
                sb=(augmentor.rand_range(rng, .97, 1.03) if do_train and do_warp and rng.rand() > 0.8 else 1),
            )
            return img[0]

        if do_train and rng.rand() >= 0.7:
            img, ld = augmentor.resize_aug(img, ld)
        if do_train:
            img = augmentor.add_noise(rng, img)
        img = get_align5p(img, ld, rng)
        if do_train and rng.rand() >= 0.5:
            img = cv2.flip(img, 1)

        return img, label

    def __getitem__(self, index: int) -> Tuple:
        path, label_meta = self.samples[index]
        ld = label_meta['landmark']
        label = label_meta['Labels']
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img, label = self.prepare_input(
            img, ld, label, self.rng, self.do_train, self.do_warp
        )
        return img, label_meta

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    d = DeepfakeDataset(root='./images', landmark_json="./images/ldm.json")
# vim: ts=4 sw=4 sts=4 expandtab
