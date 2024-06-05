from dataset import shuffle_dataset, split_dataset, EitlemDataLoader, EitlemDataSet
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import math
from sklearn.metrics import mean_squared_error, r2_score
import os
import torch
import shutil
import json
import sys
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch import autocast
import re
import random
scaler = GradScaler()


class Tester(object):
    def __init__(self, device, loss_fn, log10=False):
        self.device = device
        self.loss_fn = loss_fn
        self.log10 = log10
        self.saved_file_path = None

    def test(self, model, loader, N, desc):
        testY, testPredict = [], []
        loss_total = 0
        model.eval()
        with torch.no_grad():
            for data in tqdm(loader, desc=desc, leave=False):
                pre_value = model(data.to(self.device))
                if self.loss_fn is not None:
                    loss_total += self.loss_fn(pre_value, data.value).item()
                testY.extend(data.value.cpu().tolist())
                testPredict.extend(pre_value.cpu().tolist())
            if not self.log10:
                testY = np.log10(np.power(2, testY))
                testPredict = np.log10(np.power(2, testPredict))
            else:
                testY = np.array(testY)
                testPredict = np.array(testPredict)
            MAE = np.abs(testY - testPredict).sum() / N
            rmse = np.sqrt(mean_squared_error(testY, testPredict))
            r2 = r2_score(testY, testPredict)
            return MAE, rmse, r2, loss_total/N

    def save_model(self, model, file_path):
        torch.save(model.state_dict(), file_path)
        if self.saved_file_path is not None:
            os.remove(self.saved_file_path)
        self.saved_file_path = file_path

class Trainer(object):
    def __init__(self, device, loss_fn, log10=False):
        self.device = device
        self.loss_fn = loss_fn
        self.log10 = log10
    def run(self, model, loader, optimizer, N, desc):
        model.train()
        loss_train = 0
        testY, testPredict = [], []
        i = 0
        for data in tqdm(loader, desc=desc, leave=False):
            i += 1
            optimizer.zero_grad()
#             with autocast(device_type='cuda', dtype=torch.float16):
            pre_value = model(data.to(self.device))
            loss = self.loss_fn(pre_value, data.value)
#             scaler.scale(loss).backward()
            loss.backward()
#             scaler.step(optimizer)
            optimizer.step()
#             scaler.update()
            loss_train += loss.item()
            testY.extend(data.value.cpu().tolist())
            testPredict.extend(pre_value.cpu().tolist())
        if not self.log10:
            testY = np.log10(np.power(2, testY))
            testPredict = np.log10(np.power(2, testPredict))
        else:
            testY = np.array(testY)
            testPredict = np.array(testPredict)
        train_MAE = np.abs(testY - testPredict).sum() / N
        train_rmse = np.sqrt(mean_squared_error(testY, testPredict))
        train_r2 = r2_score(testY, testPredict)
        return train_MAE, train_rmse, train_r2, loss_train/N

def get_pair_info(dataset_path, Type, infer=True):
    train_pair = torch.load(os.path.join(dataset_path, Type, f"{Type}TrainPairInfo"))
    test_pair = torch.load(os.path.join(dataset_path, Type, f"{Type}TestPairInfo"))
    if infer:
        extra_info = torch.load(os.path.join(dataset_path, Type, f"extraInfo"))
        train_pair.extend(extra_info[:int(0.9*len(extra_info))])
        test_pair.extend(extra_info[int(0.9*len(extra_info)):])
    return train_pair, test_pair
    
def fold_test_split(dataset_path, Type, fold):
    pair_info = torch.load(os.path.join(dataset_path, Type, f"{Type}TrainPairInfo")) + torch.load(os.path.join(dataset_path, Type, f"{Type}TestPairInfo"))
    n = len(pair_info) // 5
    test_set = pair_info[fold*n:(fold+1)*n]
    return pair_info[0:fold*n] + pair_info[(fold+1)*n:], test_set