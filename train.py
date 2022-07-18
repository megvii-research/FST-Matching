#!/usr/bin/env python3
import argparse
import os

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
    parser.add_argument('--model_type', type=str, help='The type of the model.', choices=['source', 'target', 'det', 'FSTMatching'])
    parser.add_argument('--backbone', type=str, help='The backbone of the model.', default='res18')
    parser.add_argument('--batch_size', type=int, help='The batch size of the training set.', default=128)
    parser.add_argument('--epoch_num', type=int, help='Total number of epochs for training.', default=200)
    parser.add_argument('--save_path', type=str, help='The path to the saved checkpoints.', default='./checkpoints')
    args = parser.parse_args()
    return args


def save_checkpoint(model, save_path, model_type, epoch_num):
    save_path = os.path.join(save_path, model_type)
    os.makedirs(save_path, exist_ok=True)
    megengine.save(model.state_dict(), f'{save_path}/{epoch_num}.pkl')


@dist.launcher
def train():
    args = args_func()
    # get devices.
    rank = dist.get_rank()

    cpu_num = megengine.get_device_count('cpu')
    gpu_num = megengine.get_device_count('gpu')

    if rank == 0:
        print(f"MegEngine cuda avalible {megengine.is_cuda_available()}")
        print(f"MegEngine detect {cpu_num} CPU")
        print(f"MegEngine detect {gpu_num} GPU")

    device_num = gpu_num if gpu_num else cpu_num

    # init model.
    model = load_model(args.model_type, args.backbone)
    # model parallel.
    dist.bcast_list_(model.parameters())

    # optimizer init.
    gm = autodiff.GradManager()
    gm.attach(model.tensors(), callbacks=[dist.make_allreduce_cb("sum")])
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=4e-3)

    print(f"load deepfake dataset from {args.dataset}..")
    train_dataset = DeepfakeDataset(args.dataset, args.ld, args.model_type, mode='train')
    train_sampler = data.RandomSampler(train_dataset, batch_size=args.batch_size)

    # TODO: add align 5p and add noise augmentor.
    transform = T.Compose([
        T.ToMode("CHW"),
    ])

    train_dataloader = data.DataLoader(train_dataset, train_sampler, transform)

    # start trining.
    for epoch in range(args.epoch_num):
        model.train()

        for batch_data, batch_label in train_dataloader:

            lr = 1e-5
            optimizer.learning_rate = lr


            # convert ndarray to megengine tensor.
            batch_data = megengine.Tensor(batch_data)
            labels = megengine.Tensor(batch_label['Labels'])
            src_labels = megengine.Tensor(batch_label['SrcLabels'])
            tar_labels = megengine.Tensor(batch_label['TarLabels'])

            # split data by range. for data parellel.
            size = batch_data.shape[0] // device_num
            l = size * rank
            r = min(size * (rank + 1), batch_data.shape[0])
            batch_data, labels = batch_data[l:r], labels[l:r]
            tar_labels = tar_labels[l:r]
            src_labels = src_labels[l:r]
            with gm:
                if args.model_type == 'source' :
                    srcRes = model(batch_data)
                    srcLoss = F.nn.cross_entropy(srcRes, src_labels)
                    loss = srcLoss
                elif args.model_type == 'target':
                    tarRes = model(batch_data)
                    tarLoss = F.nn.cross_entropy(tarRes, tar_labels)
                    loss = tarLoss
                elif args.model_type == 'det':
                    outputs = model(batch_data)
                    detLoss = F.nn.cross_entropy(outputs, labels)
                    loss = detLoss
                elif args.model_type == 'FSTMatching':
                    outputs, srcRes, tarRes, interactionRes = model(batch_data)
                    detLoss = F.nn.cross_entropy(outputs, labels)
                    srcLoss = F.nn.cross_entropy(srcRes, src_labels) * 0.5
                    tarLoss = F.nn.cross_entropy(tarRes, tar_labels)
                    interactionLoss = -(interactionRes[:, labels] - interactionRes[:, 1-labels]).sum() * 1e2
                    loss = detLoss + srcLoss + tarLoss + interactionLoss
                else:
                    ValueError("Unsupported type of models!")
                gm.backward(loss)

            loss = dist.functional.all_reduce_sum(loss) / dist.get_world_size()

            optimizer.step().clear_grad()

            if rank != 0:
                continue
            outputs = [
                "e:{},{}/".format(epoch, args.batch_size),
                "loss {:.8f} ".format(loss.item()),
                "lr:{:.4g}".format(lr),
            ]
            print("".join(outputs))
        if rank != 0:
            continue
        save_checkpoint(model, args.save_path, args.model_type, f'{args.backbone}_epoch{epoch}')


if __name__ == "__main__":
    train()

# vim: ts=4 sw=4 sts=4 expandtab
