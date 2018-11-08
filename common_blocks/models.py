import torchvision
import torch
from torch import nn
import os
from src import utils


def resnet101(num_classes=6, model_save_path=None):
    logger = utils.get_logger('tiangong')
    model = torchvision.models.resnet101(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if model_save_path is not None and os.path.exists(model_save_path):
        logger.info('using cached model weights.')
        model.load_state_dict(torch.load(model_save_path))
    for param in model.parameters():
        param.requires_grad = False
    return model
