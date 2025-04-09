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

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        参数:
            csv_file (str): CSV 文件路径，包含图片名称和标签。
            img_dir (str): 图片文件夹路径。
            transform (callable, optional): 图片预处理函数。
        """
        self.labels_df = pd.read_csv(csv_file, header=None)  # 读取 CSV 文件
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)  # 返回数据集大小

    def __getitem__(self, idx):
        # 获取图片名称和标签
        img_name = self.labels_df.iloc[idx, 0]  # 第一列是图片名称
        if "virus" in img_name.lower():
            label = torch.Tensor([0, 1, 0])  # virus
        elif "bacteria" in img_name.lower():
            label = torch.Tensor([0, 0, 1])  # bacteria
        else:
            label = torch.Tensor([1, 0, 0])  # 其他
        # 加载图片
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')  # 转换为 RGB 格式

        # 应用预处理
        if self.transform:
            image = self.transform(image)

        return torch.Tensor(image), label


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
    # 创建训练集 Logger
    logger_train = logging.getLogger('train_logger')
    logger_train.setLevel(logging.INFO)
    # 创建文件处理器并设置格式
    file_handler_train = logging.FileHandler(log_file_train, mode='w')
    file_handler_train.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    # 将处理器添加到 logger
    logger_train.addHandler(file_handler_train)

    # 创建测试集日志目录和文件
    os.makedirs(log_dir, exist_ok=True)
    log_file_test = os.path.join(log_dir, "test_log.txt")
    # 创建测试集 Logger
    logger_test = logging.getLogger('test_logger')
    logger_test.setLevel(logging.INFO)
    # 创建文件处理器并设置格式
    file_handler_test = logging.FileHandler(log_file_test, mode='w')
    file_handler_test.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    # 将处理器添加到 logger
    logger_test.addHandler(file_handler_test)
    num_classes = args.CLS
    irene =  GLAVit(
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
    ).cuda()

    criterion = CrossEntropyLoss().cuda()
    # irene = IRENE(config, 224, zero_head=True, num_classes=num_classes, isToken=args.IS_TOKEN).cuda()
    optimizer = optim.AdamW(irene.parameters(), lr=3e-5)
    scaler = amp.GradScaler()
    lf = lambda x: ((1 + math.cos(x * math.pi / num_epochs)) / 2) * (1 - 3e-7) + 3e-7  # cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    train_data = CustomImageDataset(args.TRAIN_DIR, args.TRAIN_IMG_DIR, transform=data_transforms['train'])
    trainloader = DataLoader(train_data, batch_size=args.BSZ, shuffle=True, num_workers=4,
                             pin_memory=False)
    test_data = CustomImageDataset(args.TEST_DIR, args.TEST_IMG_DIR, transform=data_transforms['test'])
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

        irene.train()
        for data in tqdm(trainloader):
            optimizer.zero_grad()
            imgs, labels = data
            imgs = imgs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            with amp.autocast():
                logits = irene(imgs)
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

        irene.eval()
        with torch.no_grad():
            for data in tqdm(testloader):
                imgs, labels = data
                imgs = imgs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                preds = irene(imgs)

                all_preds.extend(torch.argmax(preds, dim=1).cpu().numpy())
                all_labels.extend(torch.argmax(labels, dim=1).cpu().numpy())
                all_probs.extend(F.softmax(preds, dim=1).cpu().numpy())

        # 统一在整个数据集上计算指标
        # train & val
        # auroc = roc_auc_score(all_labels, all_probs, average=None, multi_class='ovr')
        conf_matrix_train = confusion_matrix(all_labels_train, all_preds_train)
        # conf_matrix_val = confusion_matrix(all_labels_val, all_preds_val)

        # 每个类别的accuracy
        train_accuracy_per_class = conf_matrix_train.diagonal() / conf_matrix_train.sum(axis=1)
        # val_accuracy_per_class = conf_matrix_val.diagonal() / conf_matrix_val.sum(axis=1)

        # 总体的accuracy
        train_overall_accuracy = accuracy_score(all_labels_train, all_preds_train)
        # val_overall_accuracy = accuracy_score(all_labels_val, all_preds_val)

        avg_train_precision = precision_score(all_labels_train, all_preds_train, average='macro')
        avg_train_precision_n = precision_score(all_labels_train, all_preds_train, average=None)
        avg_train_precision_w = precision_score(all_labels_train, all_preds_train, average='weighted')
        # avg_val_precision = precision_score(all_labels_val, all_preds_val, average='macro')
        # avg_val_precision_n = precision_score(all_labels_val, all_preds_val, average=None)
        # avg_val_precision_w = precision_score(all_labels_val, all_preds_val, average='weighted')
        avg_train_recall = recall_score(all_labels_train, all_preds_train, average='macro')
        avg_train_recall_n = recall_score(all_labels_train, all_preds_train, average=None)
        avg_train_recall_w = recall_score(all_labels_train, all_preds_train, average='weighted')
        # avg_val_recall = recall_score(all_labels_val, all_preds_val, average='macro')
        # avg_val_recall_n = recall_score(all_labels_val, all_preds_val, average=None)
        # avg_val_recall_w = recall_score(all_labels_val, all_preds_val, average='weighted')
        avg_train_f1 = f1_score(all_labels_train, all_preds_train, average='macro')
        avg_train_f1_n = f1_score(all_labels_train, all_preds_train, average=None)
        avg_train_f1_w = f1_score(all_labels_train, all_preds_train, average='weighted')
        # avg_val_f1 = f1_score(all_labels_val, all_preds_val, average='macro')
        # avg_val_f1_n = f1_score(all_labels_val, all_preds_val, average=None)
        # avg_val_f1_w = f1_score(all_labels_val, all_preds_val, average='weighted')
        balanced_acc_train = balanced_accuracy_score(all_labels_train, all_preds_train)
        mcc_train = matthews_corrcoef(all_labels_train, all_preds_train)
        # balanced_acc_val = balanced_accuracy_score(all_labels_val, all_preds_val)
        # mcc_val = matthews_corrcoef(all_labels_val, all_preds_val)
        # avg_train_auroc = roc_auc_score(all_labels_train, all_probs_train, average='macro', multi_class='ovr')
        # avg_val_auroc = roc_auc_score(all_labels_val, all_probs_val, average='macro', multi_class='ovr')

        # test
        precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
        recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
        f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
        auroc = roc_auc_score(all_labels, all_probs, average=None, multi_class='ovr')
        conf_matrix = confusion_matrix(all_labels, all_preds)
        precision_m = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall_m = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_m = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        auroc_m = roc_auc_score(all_labels, all_probs, average='macro', multi_class='ovr')
        precision_w = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall_w = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1_w = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        auroc_w = roc_auc_score(all_labels, all_probs, average='weighted', multi_class='ovr')
        # 每个类别的accuracy
        test_accuracy_per_class = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        # 总体的accuracy
        test_overall_accuracy = accuracy_score(all_labels, all_preds)
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        mcc = matthews_corrcoef(all_labels, all_preds)
        max_acc = max(max_acc, test_overall_accuracy)

        # train & val
        print(f'epoch = {epoch}')
        print(f"Train Confusion Matrix epoch {epoch}:\n{conf_matrix_train}")
        # print(f"Val Confusion Matrix epoch {epoch}:\n{conf_matrix_val}")
        print(f"Train - Loss: {train_loss / len(trainloader):.4f}")
        print(
            f'Accuracy_per_class_train: {train_accuracy_per_class}, Accuracy_all_train: {train_overall_accuracy}')
        print(
            f'None: Precision: {avg_train_precision:.4f}, Recall: {avg_train_recall:.4f}, F1: {avg_train_f1:.4f}')
        print(f"macro: Precision: {avg_train_precision_n}, Recall: {avg_train_recall_n}, F1: {avg_train_f1_n}")
        print(
            f"weighted: Precision: {avg_train_precision_w}, Recall: {avg_train_recall_w}, F1: {avg_train_f1_w}")
        print(f'balanced_acc: {balanced_acc_train}, mcc: {mcc_train}')
        # print(f"Val - Loss: {val_loss / len(valloader):.4f}")
        # print(f'Accuracy_per_class_val: {val_accuracy_per_class}, Accuracy_all_val: {val_overall_accuracy}')
        # print(f'None: Precision: {avg_val_precision:.4f}, Recall: {avg_val_recall:.4f}, F1: {avg_val_f1:.4f}')
        # print(f"macro: Precision: {avg_val_precision_n}, Recall: {avg_val_recall_n}, F1: {avg_val_f1_n}")
        # print(f"weighted: Precision: {avg_val_precision_w}, Recall: {avg_val_recall_w}, F1: {avg_val_f1_w}")
        # (f'balanced_acc: {balanced_acc_val}, mcc: {mcc_val}')
        logger_train.info(f'epoch = {epoch}')
        logger_train.info(f"Train Confusion Matrix epoch {epoch}:\n{conf_matrix_train}")
        # logger_train.info(f"Val Confusion Matrix epoch {epoch}:\n{conf_matrix_val}")
        logger_train.info(f"Train - Loss: {train_loss / len(trainloader):.4f}")
        logger_train.info(
            f'Accuracy_per_class_train: {train_accuracy_per_class}, Accuracy_all_train: {train_overall_accuracy}')
        logger_train.info(
            f'None: Precision: {avg_train_precision:.4f}, Recall: {avg_train_recall:.4f}, F1: {avg_train_f1:.4f}')
        logger_train.info(
            f"macro: Precision: {avg_train_precision_n}, Recall: {avg_train_recall_n}, F1: {avg_train_f1_n}")
        logger_train.info(
            f"weighted: Precision: {avg_train_precision_w}, Recall: {avg_train_recall_w}, F1: {avg_train_f1_w}")
        logger_train.info(f'balanced_acc: {balanced_acc_train}, mcc: {mcc_train}')
        # logger_train.info(f"Val - Loss: {val_loss / len(valloader):.4f}")
        # logger_train.info(
        #     f'Accuracy_per_class_val: {val_accuracy_per_class}, Accuracy_all_val: {val_overall_accuracy}')
        # logger_train.info(
        #     f'None: Precision: {avg_val_precision:.4f}, Recall: {avg_val_recall:.4f}, F1: {avg_val_f1:.4f}')
        # logger_train.info(
        #     f"macro: Precision: {avg_val_precision_n}, Recall: {avg_val_recall_n}, F1: {avg_val_f1_n}")
        # logger_train.info(
        #     f"weighted: Precision: {avg_val_precision_w}, Recall: {avg_val_recall_w}, F1: {avg_val_f1_w}")
        # logger_train.info(f'balanced_acc: {balanced_acc_val}, mcc: {mcc_val}')

        # test
        print(f"Confusion Matrix:\n{conf_matrix}")
        print(
            f"Precision (None): {precision}  Precision (macro): {precision_m}  Precision (weighted): {precision_w}")
        print(f"Recall (None): {recall}  Recall (macro): {recall_m}  Recall (weighted): {recall_w}")
        print(f"F1 Score (None): {f1}  F1 Score (macro): {f1_m}  F1 Score (weighted): {f1_w}")
        print(f"AUROC (None): {auroc}  AUROC (macro): {auroc_m}  AUROC (weighted): {auroc_w}")
        print(f'Accuracy_per_class: {test_accuracy_per_class}, Accuracy_all: {test_overall_accuracy}')
        print(f'balanced_acc: {balanced_acc}, mcc: {mcc}')
        print(f'best_acc: {max_acc}')
        logger_test.info(f'epoch = {epoch}')
        logger_test.info(f"Confusion Matrix:\n{conf_matrix}")
        logger_test.info(
            f"Precision (None): {precision}  Precision (macro): {precision_m}  Precision (weighted): {precision_w}")
        logger_test.info(f"Recall (None): {recall}  Recall (macro): {recall_m}  Recall (weighted): {recall_w}")
        logger_test.info(f"F1 Score (None): {f1}  F1 Score (macro): {f1_m}  F1 Score (weighted): {f1_w}")
        logger_test.info(f"AUROC (None): {auroc}  AUROC (macro): {auroc_m}  AUROC (weighted): {auroc_w}")
        logger_test.info(
            f'Accuracy_per_class: {test_accuracy_per_class}, Accuracy_all: {test_overall_accuracy}')
        logger_test.info(f'balanced_acc: {balanced_acc}, mcc: {mcc}')
        logger_test.info(f'best_acc: {max_acc}')

        train_accuracies.append(train_overall_accuracy)
        test_accuracies.append(test_overall_accuracy)

if __name__ == '__main__':
    server = dict()
    server['lab'] = {
        'default_img_train': r'',
        'default_img_test': r'',
        'default_text_path': r'',
        'log_dir': r''
    }
    default_path = server['lab']

    parser = argparse.ArgumentParser(description="Training and Testing Script")

    # 添加模式参数,train包含了test
    parser.add_argument('--mode', action='store', dest='mode', required=True, type=str, choices=['train', 'test'],
                        help="Mode: train or test")

    # 分类数、batch size等参数
    parser.add_argument('--CLS', action='store', dest='CLS', required=True, type=int)  # 分类数
    parser.add_argument('--BSZ', action='store', dest='BSZ', required=True, type=int)  # batch size
    parser.add_argument('--IS_TOKEN', action='store', dest='IS_TOKEN', type=bool,
                        help='use cls token or not', default=True)  # 是否使用cls

    parser.add_argument('--TRAIN_IMG_DIR', action='store', dest='TRAIN_IMG_DIR', type=str,
                        help='Train image directory', default=default_path['default_img_train'])  # 训练图像目录
    parser.add_argument('--TEST_IMG_DIR', action='store', dest='TEST_IMG_DIR', type=str,
                        help='TEST image directory', default=default_path['default_img_test'])  # 训练图像目录

    default_text_dir = default_path['default_text_path']
    parser.add_argument('--TRAIN_DIR', action='store', dest='TRAIN_DIR', type=str, help='Train data directory',
                        default=os.path.join(default_text_dir, 'train.csv'))  # 训练文本路径
    parser.add_argument('--TEST_DIR', action='store', dest='TEST_DIR', type=str, help='Test data directory',
                        default=os.path.join(default_text_dir, 'test.csv'))  # 测试文本路径

    args = parser.parse_args()
    log_dir = default_path['log_dir']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    num_epochs = 120
    train_accuracies = []
    test_accuracies = []

    # 根据模式选择训练或测试
    if args.mode == 'train':
        train(args)
