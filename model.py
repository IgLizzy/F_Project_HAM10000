from torch import nn
from utils import *
import torch.nn.functional as F
from math import sqrt
from itertools import product as product
import torchvision
from torchvision import models,transforms

from utils import set_parameter_requires_grad

import segmentation_models_pytorch as smp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_model(model_name: str, num_classes: int, feature_extract: bool, use_pretrained: bool = True) -> torch.nn.Module:

    """
    Инициализирует предобученную модель.
    
    Параметры:
        model_name (str): Название архитектуры ('resnet', 'vgg11', 'vgg13', 'densenet')
        num_classes (int): Количество классов для классификации
        feature_extract (bool): Если True, замораживает веса модели (кроме последнего слоя)
        use_pretrained (bool): Использовать предобученные веса (по умолчанию True)
    
    Возвращает:
        torch.nn.Module: Инициализированная модель
    
    Исключения:
        ValueError: При указании недопустимого имени модели
    """
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ resnet101
        """
        model_ft = models.resnet101(weights=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "vgg11":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(weights=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "vgg13":
        """ VGG16_bn
        """
        model_ft = models.vgg13_bn(weights=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)


    elif model_name == "densenet":
        """ Densenet121
        """
        model_ft = models.densenet121(weights=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)

    else:
        print("Invalid model name, exiting...")
        exit()
    return model_ft



def initialize_segment_model(model_name: str) -> torch.nn.Module:
    """
    Инициализирует модель для семантической сегментации
    
    Параметры:
        model_name: Название архитектуры ('DeepLabV3' или 'Unet')
    
    Возвращает:
        Модель для семантической сегментации
    
    """
    model_ft = None
    input_size = 0

    if model_name == "DeepLabV3":
        """
        deeplabv3_resnet50
        """
        model_ft = smp.DeepLabV3(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)

    elif model_name == "Unet":
        '''
        classic Unet
        '''
        model_ft = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
        
    else:
        print("Invalid model name, exiting...")
        exit()
    return model_ft


    