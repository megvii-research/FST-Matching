#!/usr/bin/env python3
import argparse
from sklearn.metrics import roc_auc_score

import megengine
import megengine.functional as F
import megengine.optimizer as optim
import megengine.autodiff as autodiff
import megengine.distributed as dist

import megengine.data as data
import megengine.data.transform as T

from backbones.model import load_model
from dataset.deepfake_dataset import DeepfakeDataset
from lib.util import load_config


def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='The path to the dataset dir.', required=True)
    parser.add_argument('--ld', type=str, help='The path to the landmark json file of the dataset.', required=True)
    parser.add_argument('--model_type', type=str, help='The backbone of the model.', choices=['source', 'target', 'det', 'FSTMatching'])
    parser.add_argument('--backbone', type=str, help='The backbone of the model.', default='res18')
    parser.add_argument('--batch_size', type=int, help='The batch size of the training set.', default=128)
    parser.add_argument('--save_path', type=str, help='The path to the saved checkpoints.', default='./checkpoints')
    parser.add_argument('--epoch_num', type=int, help='The checkpoint to load.')
    args = parser.parse_args()
    return args


def test():
    args = args_func()

    cpu_num = megengine.get_device_count('cpu')
    gpu_num = megengine.get_device_count('gpu')

    print(f"MegEngine cuda avalible {megengine.is_cuda_available()}")
    print(f"MegEngine detect {cpu_num} CPU")
    print(f"MegEngine detect {gpu_num} GPU")

    device_num = gpu_num if gpu_num else cpu_num

    # init model.
    model = load_model(args.model_type, f'{args.backbone}_epoch{args.epoch_num}', args.save_path)
    model.eval()

    print(f"load deepfake dataset from {args.dataset}..")
    test_dataset = DeepfakeDataset(args.dataset, args.ld, args.model_type, mode='test')
    test_sampler = data.RandomSampler(test_dataset, batch_size=args.batch_size)

    transform = T.Compose([
        T.ToMode("CHW"),
    ])

    test_dataloader = data.DataLoader(test_dataset, test_sampler, transform)

    # start testing.
    pred_list = list()
    label_list = list()
    for batch_data, batch_label in test_dataloader:
        # convert ndarray to megengine tensor.
        batch_data = megengine.Tensor(batch_data)
        labels = megengine.Tensor(batch_label['Labels'])
        outputs = F.nn.softmax(model(batch_data), axis=-1)[:, 1]
        pred_list.extend(outputs.numpy().tolist())
        label_list.extend(labels.numpy().tolist())
    auc = roc_auc_score(label_list, pred_list)
    print(f"AUC of {args.dataset} is {auc:.4f}")


if __name__ == "__main__":
    test()

# vim: ts=4 sw=4 sts=4 expandtab
