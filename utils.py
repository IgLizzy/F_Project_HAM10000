import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as FT
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Назначьте больший вес миноритарным классам для функции потерь
imgs_df = pd.read_csv('HAM10000_metadata.csv')
imgs_df['cell_type_idx'] = pd.Categorical(imgs_df['dx']).codes
imgs_df['full_cell_type_name'] = imgs_df['dx'].map({'nv':'Melanocytic nevi', 'bkl':'Benign keratosis-like lesions', 
                                                    'df':'Dermatofibroma', 'mel':'Melanoma', 'vasc':'Vascular lesions', 
                                                    'bcc':'Basal cell carcinoma', 'akiec':'Actinic keratoses'})
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


def logs_to_list(df):
    '''
    функция для чтения логов
    переводть столбец из df в список метрик
    '''
    python_list = sum([ast.literal_eval(i) for i in df.to_list()], [])
    return python_list
    
def multiply_factor(df):
    '''
    функция для расчета коэфф. увеличения 
    классов при балансировке
    '''
    list_factor = []
    lbl_list = df['dx'].value_counts().to_list()
    max_lbl_count = max(lbl_list)
    for i in lbl_list:
        if i != max_lbl_count:
            list_factor.append(round(max_lbl_count/i))
    return list_factor

def tensor_to_img(tensor, shaw=False):
    '''
    преобразование объектов tensor в изображение
    необходимо для визуальной проверкт
    '''
    # Преобразуем тензор в формат (H, W, C)
    if tensor.is_cuda:
        tensor = tensor.cpu()
    image = tensor.permute(1, 2, 0).numpy()
    # Если тензор нормализован, вернем его к диапазону [0, 1]
    image = (image - image.min()) / (image.max() - image.min())
    if shaw:
        # Отображение изображения
        plt.imshow(image)
        plt.axis('off')  # Отключаем оси
        plt.show()
    else:
        return image

# feature_extract is a boolean that defines if we are finetuning or feature extracting. 
# If feature_extract = False, the model is finetuned and all model parameters are updated. 
# If feature_extract = True, only the last layer parameters are updated, the others remain fixed.
def set_parameter_requires_grad(model, feature_extracting):
    '''
    функция определяет, выполняем ли мы finetuning или feature extracting
    - если false, все параметры модели обновляются.
    - если true, обновляются только параметры последнего слоя, другие остаются фиксированными.
    '''
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_sampler_weight(dataset):
    '''
    функция для вычисления весов для сэмплера
    и последующей балансировке батчей
    '''
    targets = dataset.labels
    class_count = np.unique(targets, return_counts=True)[1]
    
    weight = 1. / class_count
    samples_weight = weight[targets]
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler

def get_class_weights(data):
    '''
    функция для расчета весов классов для функции потерь 
    используем обратный квадрат частоты, чтобы ещё больше увеличить вес миноритарных классов
    '''
    class_counts = torch.bincount(torch.tensor(data.labels))
    class_weights = (len(data.labels) ** 2) / (class_counts ** 2)
    class_weights = class_weights / class_weights.sum()
    return class_weights
# Some augmentation functions below have been adapted from
# From https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

def model_accuracy(out, lbl):
    _, pred = out.topk(1)
    pred = pred.reshape(-1, 1).float()
    lbl = lbl.reshape(-1, 1).float()
    acc = (lbl == pred).flatten().sum(dtype=torch.float32)
    return acc

#  function for semantic segmentation models
def mask_circuit(image, mask, evaluation=False, show=True):
    '''
      функция для наложения контура маски на оригинальное изображение
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
    
    return tensor_to_img(result, shaw=show)

def iou_dice_score(pred, target, multiclass=False):
    '''
    функция для определения метрик:
    - IoU (Intersection over Union);
    - Dice.
    '''
    if multiclass:
        pred = torch.sigmoid(pred) #если сегментация не бинарная
    pred = (pred>0/5).float()
    
    intersection = (pred * target).sum()
    unoin = pred.sum() + target.sum() - intersection
    
    iou = (intersection + 1e-6) / (unoin + 1e-6)
    dice = (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
    return iou.item(), dice.item()

def show_masks(images, masks, predicted_masks):
    '''
    функция для визуального сравнения исходных масок и предсказанных моделью
    вход: 
    вывод: исходное изображение, исходная маска, предсказанная маска
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
