#!/usr/bin/env python3
import argparse

from backbones.model import get_model_dict
from lib.util import load_config
from lib.data_preprocess import load_imgs
from lib.shapley import shapley_function_visual
from lib.visualization import visualize_phi, draw_result
from lib.quantitive_indicators import calculate_q, stability


def hypothesis1_verification(config):
    # get the model
    model_dict = get_model_dict(config, "hypothesis_1")
    # for each manipulation algorithm
    Manipulation_list = config['manipulation_list']
    for deepfake in Manipulation_list:
        # get the images
        imgs, img_info, labels, source_labels, target_labels = load_imgs(
            config['img_path'], deepfake, config['data_type']
        )
        # get the phi
        print('==' * 16 + 'Calculate phis' + '==' * 16)
        source_result = shapley_function_visual(
            imgs, model_dict["source"], source_labels, img_info,
            sample_times=config['default_times'],
            grid_scale=config['grid_scale']
        )
        print('==' * 16 + 'Calculate phit' + '==' * 16)
        target_result = shapley_function_visual(
            imgs, model_dict["target"], target_labels, img_info,
            sample_times=config['default_times'],
            grid_scale=config['grid_scale']
        )
        print('==' * 16 + 'Calculate phid' + '==' * 16)
        det_result = shapley_function_visual(
            imgs, model_dict["det"], labels, img_info,
            sample_times=config['default_times'],
            grid_scale=config['grid_scale']
        )
        # verify the hypothesis qualitatively
        visualize_phi(
            source_result, target_result, det_result, imgs,
            threshold=config['threshold_vis'],
            save_path=config['hypothesis_1']['save_dir']
        )
        # verify the hypothesis quantatively
        Q = calculate_q(
            source_result, target_result, det_result,
            threshold_interval=config['threshold_interval'],
            threshold_step_size=config['threshold_step_size'],
        )
        print('==' * 16 + f'{deepfake}:' + '==' * 16)
        print(f"Source backbone: {config['source_backbone']}")
        print(f"Target backbone: {config['target_backbone']}")
        print(f"Det backbone: {config['det_backbone']}")
        print(f'Q: {Q}')


def hypothesis2_verification(config):
    # Save_Q_dict
    Q_reult_dict = dict()
    # get the model
    model_dict = get_model_dict(config, "hypothesis_2")
    # for each manipulation algorithm
    Manipulation_list = config['manipulation_list']
    for deepfake in Manipulation_list:
        # get the images
        imgs, img_info, labels, source_labels, target_labels = load_imgs(
            config['img_path'], deepfake, config['data_type']
        )
        # get the phi
        print('==' * 16 + 'Calculate phis' + '==' * 16)
        source_result = shapley_function_visual(
            imgs, model_dict["source"], source_labels, img_info,
            sample_times=config['default_times'],
            grid_scale=config['grid_scale']
        )
        print('==' * 16 + 'Calculate phit' + '==' * 16)
        target_result = shapley_function_visual(
            imgs, model_dict["target"], target_labels, img_info,
            sample_times=config['default_times'],
            grid_scale=config['grid_scale']
        )
        print('==' * 16 + 'Calculate phid_pair' + '==' * 16)
        det_pair_result = shapley_function_visual(
            imgs, model_dict["det_pair"], labels, img_info,
            sample_times=config['default_times'],
            grid_scale=config['grid_scale']
        )
        print('==' * 16 + 'Calculate phid_unpair' + '==' * 16)
        det_unpair_result = shapley_function_visual(
            imgs, model_dict["det_unpair"], labels, img_info,
            sample_times=config['default_times'],
            grid_scale=config['grid_scale']
        )
        # verify the hypothesis quantatively
        Q_pair = calculate_q(
            source_result, target_result, det_pair_result,
            threshold_interval=config['threshold_interval'],
            threshold_step_size=config['threshold_step_size'],
            return_dict=True
        )
        Q_unpair = calculate_q(
            source_result, target_result, det_unpair_result,
            threshold_interval=config['threshold_interval'],
            threshold_step_size=config['threshold_step_size'],
            return_dict=True
        )
        Q_reult_dict[deepfake] = {key: [val, Q_unpair[key]] for key, val in Q_pair.items()}
    print('==' * 16 + 'Draw the result' + '==' * 16)
    print(f"Source backbone: {config['source_backbone']}")
    print(f"Target backbone: {config['target_backbone']}")
    print(f"Det backbone: {config['det_backbone']}")
    draw_result(Q_reult_dict, config['hypothesis_2']['save_dir'])


def hypothesis3_verification(config):
    # get the model
    model_dict = get_model_dict(config, "hypothesis_3")
    # for each manipulation algorithm
    Manipulation_list = config['manipulation_list']
    for model_type, model in model_dict.items():
        for deepfake in Manipulation_list:
            # get the images
            imgs_raw, img_raw_info, labels, source_labels, target_labels = load_imgs(
                config['img_path'], deepfake, "raw"
            )
            imgs_c23, img_c23_info, _, _, _ = load_imgs(
                config['img_path'], deepfake, "c23"
            )
            imgs_c40, img_c40_info, _, _, _ = load_imgs(
                config['img_path'], deepfake, "c40"
            )

            # get the phi
            print('==' * 16 + 'Calculate phi on raw' + '==' * 16)
            raw_result = shapley_function_visual(
                imgs_raw, model, labels, img_raw_info,
                sample_times=config['default_times'],
                grid_scale=config['grid_scale']
            )
            print('==' * 16 + 'Calculate phi on c23' + '==' * 16)
            c23_result = shapley_function_visual(
                imgs_c23, model, labels, img_c23_info,
                sample_times=config['default_times'],
                grid_scale=config['grid_scale']
            )
            print('==' * 16 + 'Calculate phi on c40' + '==' * 16)
            c40_result = shapley_function_visual(
                imgs_c40, model, labels, img_c40_info,
                sample_times=config['default_times'],
                grid_scale=config['grid_scale']
            )
            # verify the hypothesis quantatively
            delta = stability(raw_result, c23_result, c40_result)
            print('==' * 16 + f'{deepfake}:' + '==' * 16)
            print(f"{model_type} backbone:", config[f"{model_type.split('_')[0]}_backbone"])
            print(f'delta: {delta}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str,
        help='The path of the config file.', required=True)
    parser.add_argument(
        '--hypothesis_number', type=int,
        help='The number of hypothesis.', required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    assert args.hypothesis_number in [1, 2, 3]

    if args.hypothesis_number == 1:
        hypothesis1_verification(config)
    elif args.hypothesis_number == 2:
        hypothesis2_verification(config)
    else:
        hypothesis3_verification(config)

# vim: ts=4 sw=4 sts=4 expandtab
