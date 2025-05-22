import json
import ast
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Union, Literal, List, TypeVar, Tuple, Optional
import matplotlib.pyplot as plt
from PIL import Image
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset, WeightedRandomSampler
import random
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as FT
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Назначьте больший вес миноритарным классам для функции потерь
imgs_df = pd.read_csv('HAM10000_metadata.csv')
mod_imgs_df = pd.read_csv('full_isic.csv')
imgs_df['cell_type_idx'] = pd.Categorical(imgs_df['dx']).codes
imgs_df['full_cell_type_name'] = imgs_df['dx'].map({'nv':'Melanocytic nevi', 'bkl':'Benign keratosis-like lesions', 
                                                    'df':'Dermatofibroma', 'mel':'Melanoma', 'vasc':'Vascular lesions', 
                                                    'bcc':'Basal cell carcinoma', 'akiec':'Actinic keratoses'})

mod_imgs_df['dx'] = mod_imgs_df['diagnosis'].map({'nevus':'nv', 'benign keratosis-like lesions':'bkl', 
                                                    'dermatofibroma':'df', 'melanoma':'mel', 'vascular lesions':'vasc', 
                                                    'basal cell carcinoma':'bcc', 'actinic keratoses':'akiec'})
mod_imgs_df['cell_type_idx'] = pd.Categorical(mod_imgs_df['dx']).codes
# список multiply_factor содержит множители для балансировки классов
# multiply_factor = [round(imgs_df['dx'].value_counts().max()/i) for i in imgs_df['dx'].value_counts().to_list() if i != imgs_df['dx'].value_counts().max()]
# список itm_list содержит классы, которые нцжно балансировать
itm_list = ['mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

root = 'C:/Users/Igor/Desktop/Project_DS50/data/'
mask_end = '_segmentation'

df_dict = {img: lbl for img, lbl in imgs_df[['dx', 'cell_type_idx']].values}
sorted_df_dict = dict(sorted(df_dict.items()))
sorted_df_list = list(sorted_df_dict.keys())
# Label map
voc_labels = ('nv', 'bkl', 'df', 'mel', 'vasc', 'bcc', 'akiec')
# 
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
# label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping


def logs_to_list(series : pd.Series) -> List[int]:
    '''
    Преобразует Series с логами в плоский список числовых метрик.

    Параметры:
        series: Series, содержащий логи

    Возвращает:
        Список числовых метрик (int или float)

    Пример:
        >>> series = pd.Series([1,2],[3,4])
        >>> logs_to_list(df)
        [1, 2, 3, 4]
    '''
    python_list = sum([ast.literal_eval(i) for i in series.to_list()], [])
    return python_list
    
def multiply_factor(df : pd.DataFrame, target_column : str, coef:int = 1) -> Union[List[int], List[str]]:
    '''
    Вычисляет коэффициенты увеличения для балансировки классов.

    Параметры:
        df: DataFrame с данными

    Возвращает:
        кортеж со списком коэффциентов для баланса классов и список классов для балансировки

    Примеры:
         list_factor, lbl_list = multiply_factor(df)
         list_factor -> списком коэффциентов для минорных классов
         lbl_list -> список классов для балансировки
    '''
    list_factor = []
    lbl_list = []
    lbl_val_list = df[target_column].value_counts().values.tolist()
    max_lbl_count = max(lbl_val_list)
    for i, v in enumerate(lbl_val_list):
        if round(max_lbl_count/v) == 1:
            pass
        else:
            list_factor.append(round(max_lbl_count/v/coef))
            lbl_list.append(df[target_column].value_counts().index.tolist()[i])
    return list_factor, lbl_list

def tensor_to_img( tensor: Tensor, show: bool = False) -> Union[Image.Image, Tensor]:
    """
    Преобразует тензор в изображение для визуализации или возвращает преобразованный тензор.

    Параметры:
        tensor: Входной тензор в формате (C, H, W).
        show: Если True, отображает изображение с помощью matplotlib.
              Если False, возвращет тензор

    Возвращает:
        Если show = True: объект PIL.Image.Image
        Если show = Faas: тензор в формате (H, W, C)

    Примеры:
        >>> img = tensor_to_img(tensor, show=True)
        >>> tensor = tensor_to_img(tensor, show=False)
    """
    # Преобразуем тензор в формат (H, W, C)
    if tensor.is_cuda:
        tensor = tensor.cpu()
    image = tensor.permute(1, 2, 0).numpy()
    # Если тензор нормализован, вернем его к диапазону [0, 1]
    image = (image - image.min()) / (image.max() - image.min())
    if show:
        # Отображение изображения
        plt.imshow(image)
        plt.axis('off')  # Отключаем оси
        plt.show()
    else:
        return image


ModelType = TypeVar('ModelType', bound=nn.Module)
def set_parameter_requires_grad(model: ModelType, feature_extracting: bool) -> ModelType:
    '''
    Устанавливает requires_grad для параметров модели в зависимости от режима работы.

    Параметры:
        model: Модель PyTorch (nn.Module)
        feature_extracting: 
            - Если True: замораживает все параметры (режим feature extraction)
            - Если False: размораживает все параметры (режим fine-tuning)
            
    Возвращает:
        Модель с обновленными параметрами requires_grad

    Примеры:
        # Полное замораживание
        model = set_parameter_requires_grad(model, feature_extracting=True)
    '''
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_sampler_weight(dataset : Dataset) -> WeightedRandomSampler:
    '''
   Создает взвешенный семплер для балансировки классов в датасете.

    Параметры:
        dataset: Датасет, содержащий метки классов

    Возвращает:
        WeightedRandomSampler для балансировки классов

    Пример:
        >>> sampler = get_balanced_sampler(dataset)
        >>> train_loader = DataLoader(dataset, batch_size=32, sampler=sampler)
    '''
    targets = dataset.labels
    class_count = np.unique(targets, return_counts=True)[1]
    
    weight = 1. / class_count
    samples_weight = weight[targets]
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler

def get_class_weights(data : Dataset) -> Tensor:
    '''
    Вычисляет веса классов для функции потерь по обратной степени частоты.

    Параметры:
        data: Входные данные (Dataset, dict, list или Tensor с метками)
       
    Возвращает:
        Тензор с весами классов

    Примеры:
        weights = get_class_weights(dataset)
        weights = [2,3,3,1,4]
    '''
    class_counts = torch.bincount(torch.tensor(data.labels))
    class_weights = (len(data.labels) ** 2) / (class_counts ** 2)
    class_weights = class_weights / class_weights.sum()
    return class_weights


def model_accuracy(out, lbl):
    _, pred = out.topk(1)
    pred = pred.reshape(-1, 1).float()
    lbl = lbl.reshape(-1, 1).float()
    acc = (lbl == pred).flatten().sum(dtype=torch.float32)
    return acc

#  function for semantic segmentation models
def mask_circuit(image: Tensor, mask: Tensor, evaluation: bool = False, show: bool = True) -> Union[Image.Image, Tensor]:
    '''
    Накладывает контур маски на исходное изображение.

    Параметры:
        image: Исходное изображение [C, H, W] или [B, C, H, W]
        mask: Маска [H, W], [1, H, W] или [B, 1, H, W]
        evaluation: Флаг режима оценки (переносит тензоры на CPU)
        show: Флаг отображения результата

    Возвращает:
        результат работы функции передается в функцию tensor_to_img()
        и в зависимости от флага show возвращвет:
        - eсли show = True: объект PIL.Image.Image;
        - eсли show = Faas: тензор в формате (H, W, C)/ 

      '''
    if evaluation:
        image = image.cpu() #если модель обучалась на GPU тензоры необходимо копировать на CPU
        mask = mask.cpu()
    if mask.dim() != 4:
        mask = mask.unsqueeze(0)  # Теперь mask имеет размер [1, 1, 256, 256]
    
    # Ядро для нахождения границ
    kernel = torch.tensor([[-1, -1, -1],
                        [-1,  8, -1],
                        [-1, -1, -1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    # Применяем свертку для нахождения границ
    edges = F.conv2d(mask, kernel, padding=1)
    edges = (edges > 0).float()  # Бинаризация границ
    
    # Убираем batch и channel dimensions у edges
    edges = edges.squeeze(0).squeeze(0)  # Теперь edges имеет размер [256, 256]
    
    # Расширяем edges до 3 каналов для наложения на изображение
    edges = edges.unsqueeze(0).repeat(3, 1, 1)  # Теперь edges имеет размер [3, 256, 256]
    
    # Наложим контур на изображение
    result = image.clone()
    result[:, edges[0] > 0] = torch.tensor([0.0, 1.0, 0.0]).view(3, 1)
    
    return tensor_to_img(result, show=show)

def iou_dice_score(pred: Tensor, target: Tensor, threshold: float = 0.5, multiclass: bool = False) -> Tuple[Union[float, Tensor], Union[float, Tensor]]:
    '''
   Вычисляет метрики IoU (Intersection over Union) и Dice для сегментации.

    Параметры:
        pred: Предсказанные маски [N, C, H, W] или [N, H, W]
        target: Истинные маски [N, H, W] или [N, C, H, W]
        threshold: Порог бинаризации для бинарной сегментации
        multiclass: Флаг многоклассовой сегментации

    Возвращает:
        Кортеж (iou_score, dice_score)

    '''
    if multiclass:
        pred = torch.sigmoid(pred) #если сегментация не бинарная
    pred = (pred > threshold).float()
    
    intersection = (pred * target).sum()
    unoin = pred.sum() + target.sum() - intersection
    
    iou = (intersection + 1e-6) / (unoin + 1e-6)
    dice = (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
    return iou.item(), dice.item()

def show_masks(images: Union[List[Tensor], Tensor], masks: Union[List[Tensor], Tensor], predicted_masks: Union[List[Tensor], Tensor]):
    '''
    Функция для визуального сравнения исходных масок и предсказанных моделью
    
    Параметры:
        images: список исходных изображений
        masks: список истинных масок (разметка)
        predicted_masks: список масок, предсказанных моделью
    
    Возвращает:
        Визуализацию в виде: 
        1. Исходное изображение
        2. Исходная маска 
        3. Исходное изображение с контуром маски
        4. Предсказанная маска
        5. Исходное изображение с контуром предсказанной маски
    '''
    assert len(images) == len(masks) == len(predicted_masks)
    if len(images) in [1, 2]:
        fig, axes = plt.subplots(1, 5, figsize=(15,15))
        axes[0].imshow(tensor_to_img(images[0]))
        axes[0].set_xlabel('original img')
        axes[1].imshow(tensor_to_img(masks[0]))
        axes[1].set_xlabel('original mask')
        axes[2].imshow(mask_circuit(images[0], masks[0], True, False))
        axes[2].set_xlabel('original mask+img')
        axes[3].imshow(tensor_to_img(predicted_masks[0]))
        axes[3].set_xlabel('predicted mask')
        axes[4].imshow(mask_circuit(images[0], predicted_masks[0], True, False))
        axes[4].set_xlabel('predicted mask+img')
    else:
        n = np.random.randint(2, (len(images) if len(images) <= 5 else 5))
        fig, axes = plt.subplots(len(images[:n]), 5, figsize=(15,15))
        for i in range(len(images[:n])):
            axes[i, 0].imshow(tensor_to_img(images[i]))
            axes[i, 0].set_xlabel('original img')
            axes[i, 1].imshow(tensor_to_img(masks[i]))
            axes[i, 1].set_xlabel('original mask')
            axes[i, 2].imshow(mask_circuit(images[i], masks[i], True, False))
            axes[i, 2].set_xlabel('original mask+img')
            axes[i, 3].imshow(tensor_to_img(predicted_masks[i], True))
            axes[i, 3].set_xlabel('predicted mask')
            axes[i, 4].imshow(mask_circuit(images[i], predicted_masks[i], True, False))
            axes[i, 4].set_xlabel('predicted mask+img')

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
