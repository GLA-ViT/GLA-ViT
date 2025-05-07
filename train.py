from __future__ import print_function, division

import csv
import os
import sys
from collections import OrderedDict
from functools import partial
from typing import Callable, Optional, List, NamedTuple, Any
from torch.nn import CrossEntropyLoss
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch import nn
import torch.optim as optim
import pickle
from PIL import Image
from torch.cuda import amp
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, matthews_corrcoef
from torchvision.ops import Conv2dNormActivation, MLP
from torchvision.utils import _log_api_usage_once
from tqdm import tqdm
import argparse
import warnings
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import math
import logging
from models.agent_attention import AgentBlock
from models.GLAE import BlockWithCNN
from models.GLA_ViT import GLAVit
from dataset.dateset import LungDiseaseDataset, CXRDataset
if not sys.warnoptions:
    warnings.simplefilter("ignore")

def warmup(optimizer, warm_up_iters, warm_up_factor):
    def f(x):
        """根据step数返回一个学习率倍率因子, x代表step"""
        if x >= warm_up_iters:
            return 1

        alpha = float(x) / warm_up_iters
        # 迭代过程中倍率因子从warmup_factor -> 1
        return warm_up_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.25, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]),
}


def train(args):
    # 创建日志文件夹和文件
    os.makedirs(log_dir, exist_ok=True)
    log_file_train = os.path.join(log_dir, "train_log.txt")
    logger_train = logging.getLogger('train_logger')
    logger_train.setLevel(logging.INFO)
    file_handler_train = logging.FileHandler(log_file_train, mode='w')
    file_handler_train.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger_train.addHandler(file_handler_train)

    os.makedirs(log_dir, exist_ok=True)
    log_file_test = os.path.join(log_dir, "test_log.txt")
    logger_test = logging.getLogger('test_logger')
    logger_test.setLevel(logging.INFO)
    file_handler_test = logging.FileHandler(log_file_test, mode='w')
    file_handler_test.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger_test.addHandler(file_handler_test)
    
    num_classes = args.CLS
    model =  GLAVit(
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        num_classes = num_classes
    ).cuda()

    criterion = CrossEntropyLoss().cuda()
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)
    scaler = amp.GradScaler()
    lf = lambda x: ((1 + math.cos(x * math.pi / num_epochs)) / 2) * (1 - 3e-7) + 3e-7  # cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    train_data = CXRDataset(args.TRAIN_DIR, args.TRAIN_IMG_DIR, transform=data_transforms['train'])
    trainloader = DataLoader(train_data, batch_size=args.BSZ, shuffle=True, num_workers=4,
                             pin_memory=False)
    test_data = CXRDataset(args.TEST_DIR, args.TEST_IMG_DIR, transform=data_transforms['test'])
    testloader = DataLoader(test_data, batch_size=args.BSZ, shuffle=False, num_workers=4, pin_memory=False)
    max_acc = 0.0

    for epoch in range(num_epochs):

        train_loss = 0.0
        all_preds_train = []
        all_labels_train = []

        all_preds = []
        all_labels = []
        all_probs = []

        lr_scheduler = None
        if epoch == 0:
            warmup_factor = 1.0 / 1000
            warmup_iters = min(1000, len(trainloader) - 1)
            lr_scheduler = warmup(optimizer, warmup_iters, warmup_factor)

        model.train()
        for data in tqdm(trainloader):
            optimizer.zero_grad()
            imgs, labels = data
            imgs = imgs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            with amp.autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

            all_preds_train.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels_train.extend(torch.argmax(labels, dim=1).cpu().numpy())

            if lr_scheduler is not None:  # 如果使用warmup训练，逐渐调整学习率
                lr_scheduler.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            for data in tqdm(testloader):
                imgs, labels = data
                imgs = imgs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                preds = model(imgs)

                all_preds.extend(torch.argmax(preds, dim=1).cpu().numpy())
                all_labels.extend(torch.argmax(labels, dim=1).cpu().numpy())
                all_probs.extend(F.softmax(preds, dim=1).cpu().numpy())

        # 统一在整个数据集上计算指标
        # train
        conf_matrix_train = confusion_matrix(all_labels_train, all_preds_train)
        train_accuracy_per_class = conf_matrix_train.diagonal() / conf_matrix_train.sum(axis=1)
        train_overall_accuracy = accuracy_score(all_labels_train, all_preds_train)
        avg_train_precision = precision_score(all_labels_train, all_preds_train, average='macro')
        avg_train_recall = recall_score(all_labels_train, all_preds_train, average='macro')
        avg_train_f1 = f1_score(all_labels_train, all_preds_train, average='macro')
        balanced_acc_train = balanced_accuracy_score(all_labels_train, all_preds_train)
        mcc_train = matthews_corrcoef(all_labels_train, all_preds_train)

        # test
        conf_matrix = confusion_matrix(all_labels, all_preds)
        precision_m = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall_m = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_m = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        auroc_m = roc_auc_score(all_labels, all_probs, average='macro', multi_class='ovr')
        # 每个类别的accuracy
        test_accuracy_per_class = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        # 总体的accuracy
        test_overall_accuracy = accuracy_score(all_labels, all_preds)
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        mcc = matthews_corrcoef(all_labels, all_preds)
        max_acc = max(max_acc, test_overall_accuracy)

        # train
        print(f'epoch = {epoch}')
        print(f"Train Confusion Matrix epoch {epoch}:\n{conf_matrix_train}")
        print(f"Train - Loss: {train_loss / len(trainloader):.4f}")
        print(
            f'Accuracy_per_class_train: {train_accuracy_per_class}, Accuracy_all_train: {train_overall_accuracy}')
        print(f"macro: Precision: {avg_train_precision}, Recall: {avg_train_recall}, F1: {avg_train_f1}")
        print(f'balanced_acc: {balanced_acc_train}, mcc: {mcc_train}')
        logger_train.info(f'epoch = {epoch}')
        logger_train.info(f"Train Confusion Matrix epoch {epoch}:\n{conf_matrix_train}")
        logger_train.info(f"Train - Loss: {train_loss / len(trainloader):.4f}")
        logger_train.info(
            f'Accuracy_per_class_train: {train_accuracy_per_class}, Accuracy_all_train: {train_overall_accuracy}')
        logger_train.info(
            f"macro: Precision: {avg_train_precision}, Recall: {avg_train_recall}, F1: {avg_train_f1}")
        logger_train.info(f'balanced_acc: {balanced_acc_train}, mcc: {mcc_train}')

        # test
        print(f"Confusion Matrix:\n{conf_matrix}")
        print(f"Precision (macro): {precision_m}")
        print(f"Recall (macro): {recall_m}")
        print(f"F1 Score (macro): {f1_m}")
        print(f"AUROC (macro): {auroc_m}")
        print(f'Accuracy_per_class: {test_accuracy_per_class}, Accuracy_all: {test_overall_accuracy}')
        print(f'balanced_acc: {balanced_acc}, mcc: {mcc}')
        print(f'best_acc: {max_acc}')
        logger_test.info(f'epoch = {epoch}')
        logger_test.info(f"Confusion Matrix:\n{conf_matrix}")
        logger_test.info(f"Precision (macro): {precision_m}")
        logger_test.info(f"Recall (macro): {recall_m}")
        logger_test.info(f"F1 Score (None): {f1}  F1 Score (macro): {f1_m}  F1 Score (weighted): {f1_w}")
        logger_test.info(f"AUROC (macro): {auroc_m}")
        logger_test.info(
            f'Accuracy_per_class: {test_accuracy_per_class}, Accuracy_all: {test_overall_accuracy}')
        logger_test.info(f'balanced_acc: {balanced_acc}, mcc: {mcc}')
        logger_test.info(f'best_acc: {max_acc}')
