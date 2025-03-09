from torch import nn
from utils import *
import torch.nn.functional as F
from math import sqrt
from itertools import product as product
import torchvision
from torchvision import models,transforms

from utils import set_parameter_requires_grad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# this function is used during training process, to calculation the loss and accuracy
class AverageMeter(object):
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

#  function for classification models
def train_model(train_loader, model, criterion, optimizer, epoch, show):
    model.to(device)
    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    total_loss_train = []
    total_acc_train = []
    curr_iter = (epoch - 1) * len(train_loader)
    for i, data in enumerate(train_loader):
        images, labels, mask = data
        N = images.size(0)
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        prediction = outputs.max(1, keepdim=True)[1]
        train_acc.update(prediction.eq(labels.view_as(prediction)).sum().item()/N)
        train_loss.update(loss.item())
        curr_iter += 1
        if (i + 1) % show == 0:
            print(f'[epoch {epoch}], [iter {i+1}/{len(train_loader)}], [train loss {train_loss.avg:.3f}], [train acc {train_acc.avg:.3f}]')
            total_loss_train.append(train_loss.avg)
            total_acc_train.append(train_acc.avg)
        # Free memory after each epoch
        del images, labels, outputs, mask  # Delete intermediate variables
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return total_loss_train, total_acc_train


def model_test(val_loader, model, criterion, optimizer, validation=False, epoch=None):
    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    total_loss_test = []
    total_acc_test = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, labels, mask = data
            N = images.size(0)
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            prediction = outputs.max(1, keepdim=True)[1]

            val_acc.update(prediction.eq(labels.view_as(prediction)).sum().item()/N)

            val_loss.update(criterion(outputs, labels).item())
            total_loss_test.append(val_loss.avg)
            total_acc_test.append(val_acc.avg)

    if validation:
        print('------------------------------------------------------------')
        print(f'Validation [epoch {epoch}], [loss {val_loss.avg:.3f}], [acc {val_acc.avg:.3f}]')
        print('------------------------------------------------------------')
    else:
        print('------------------------------------------------------------')
        print(f'Test [loss {val_loss.avg:.3f}], [acc {val_acc.avg:.3f}]')
        print('------------------------------------------------------------')
    return total_loss_test, total_acc_test

#  function for semantic segmentation models
def segmentation_train(train_loader, model, criterion, optimizer, epoch, threshold, show):
    model.to(device)
    model.train()
    
    train_loss = AverageMeter()
    train_iou_acc = AverageMeter()
    train_dice_acc = AverageMeter()
    
    total_loss_train = []
    total_iou_acc = []
    total_dice_acc = []
    
    for i, data in enumerate(train_loader):
        images, labels, masks = data
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        prediction = (outputs > threshold).float()
        train_iou_acc.update(iou_dice_score(prediction, masks)[0])
        train_dice_acc.update(iou_dice_score(prediction, masks)[1])
        train_loss.update(loss.item())

        total_loss_train.append(train_loss.avg)
        total_iou_acc.append(train_iou_acc.avg)
        total_dice_acc.append(train_dice_acc.avg)

        if (i + 1) % show == 0:
            print(f'[epoch {epoch}], [iter {i+1}/{len(train_loader)}], [train loss {train_loss.avg:.3f}],\
[train IoU acc {train_iou_acc.avg:.3f}], [train Dice acc {train_dice_acc.avg:.3f}]')
            
        # Free memory after each epoch
        del images, labels, outputs, masks, prediction  # Delete intermediate variables
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return total_loss_train, total_iou_acc, total_dice_acc


def segmentation_eval(data_loader, model, criterion, optimizer, threshold, show=False, validation=True, epoch=None):
    model.eval()
    test_loss = AverageMeter()
    test_iou_acc = AverageMeter()
    test_dice_acc = AverageMeter()
    
    total_loss = []
    total_iou_acc = []
    total_dice_acc = []
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            images, labels, masks = data
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            predicted_masks = (outputs > threshold).float()
            loss = criterion(outputs, masks)

            test_loss.update(loss.item())
            test_iou_acc.update(iou_dice_score(predicted_masks, masks)[0])
            test_dice_acc.update(iou_dice_score(predicted_masks, masks)[1])

            total_loss.append(test_loss.avg)
            total_iou_acc.append(test_iou_acc.avg)
            total_dice_acc.append(test_dice_acc.avg)

    if validation:
        print('------------------------------------------------------------')
        print(f'Validation: [epoch {epoch}], [loss {test_loss.avg:.3f}], [IoU {test_iou_acc.avg:.3f}], [Dice {test_dice_acc.avg:.3f}]')
        print('------------------------------------------------------------')
    else:
        print('------------------------------------------------------------')
        print(f'Test: [loss {test_loss.avg:.3f}], [IoU {test_iou_acc.avg:.3f}], [Dice {test_dice_acc.avg:.3f}]')
        print('------------------------------------------------------------')
    if show:
        show_masks(images, masks, predicted_masks)
    return total_loss, total_iou_acc, total_dice_acc

