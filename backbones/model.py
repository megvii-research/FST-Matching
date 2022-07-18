import torch
from megengine import functional as F
import megengine.module as M
import megengine
from backbones.resnet import resnet18
from backbones.fstmatching import FSTNet
import pickle


def get_model_dict(config, hypothesis):
    """
    Get a list of models
    :param config: dict of configuration
    :param hypothesis: hypothesis1/2/3
    :return: model_dict, e.g. {'source': ResNet18}
    """
    model_dict = dict()
    # get the model list
    model_list = config[hypothesis]['model_list']
    for model_type in model_list:
        model = load_model(
            model_type=model_type,
            model_backbone=config[f'{model_type.split("_")[0]}_backbone'],
            model_path=config['model_path'],
        )
        model.eval()
        model_dict[model_type] = model
    return model_dict


def load_model(model_type, model_backbone, model_path=None):
    """
    load one model
    :param model_path: ./models
    :param model_type: source/target/det
    :param model_backbone: res18/res34/Efficient
    :param use_cuda: True/False
    :return: model
    """
    if model_path:
        pretrained_model = f"{model_path}/{model_type}/{model_backbone}.pkl"
    else:
        pretrained_model = None
    
    if model_backbone.startswith("res18"):
        if model_type in ['source', 'target']:
            model = resnet18(pretrained=True)
        elif model_type.startswith("det"):
            model = resnet18(pretrained=True)
            num_ftrs = model.out_num_features
            model.fc = M.Linear(num_ftrs, 2)
        elif model_type == 'FSTMatching':
            model = FSTNet()
        else:
            raise ValueError("Unsupported type of models!")
    else:
        raise ValueError("Unsupported backbone of models!")

    if pretrained_model:
        checkpoint = megengine.load(pretrained_model)
        model.load_state_dict(checkpoint)
    return model

