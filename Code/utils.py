"""
File: utils.py
Author: Xiaowei Shen
Email: XiaoweiShen@buct.edu.cn
Created: 2023-11-29
Description: .

Copyright (c) 2023 Xiaowei Shen
This code is licensed under the MIT License.
See the LICENSE file for details.
"""
from dataset import shuffle_dataset, split_dataset, CustomDataLoader, CustomDataSet, ESMUniCustomDataset
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from queue import PriorityQueue
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random
import shutil
import torch
import json
import math
import sys
import os
import re

class Tester(object):
    def __init__(self, device, loss_fn, top=11, log10=False):
        self.device = device
        self.q = PriorityQueue(top)
        self.loss_fn = loss_fn
        self.top = top
        self.log10 = log10
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

    def save_model(self, model, filename, performance, save=False):
        if save:
            torch.save(model.state_dict(), filename)
            self.q.put((0, filename))
        elif self.q.qsize() < self.top-1:
            self.q.put((performance, filename))
            torch.save(model.state_dict(), filename)
        else:
            item = self.q.get()
            if performance > item[0]:
                self.q.put((performance, filename))
                os.remove(item[1])
                torch.save(model.state_dict(), filename)
            else:
                self.q.put(item)
class Trainer(object):
    def __init__(self, device, loss_fn, log10=False):
        self.device = device
        self.loss_fn = loss_fn
        self.log10 = log10
    def run(self, model, loader, optimizer, N, desc):
        model.train()
        loss_train = 0
        testY, testPredict = [], []
        for data in tqdm(loader, desc=desc, leave=False):
            optimizer.zero_grad()
            pre_value = model(data.to(self.device))
            loss = self.loss_fn(pre_value, data.value)
            loss.backward()
            optimizer.step()
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
    
def get_pair_info(path):
    if os.path.exists(os.path.join(path, "train_info")) and os.path.exists(os.path.join(path, "test_info")):
        train_info = torch.load(os.path.join(path, "train_info"))
        tempPair = torch.load(os.path.join(path, "test_info"))
        valid_info, test_info = split_dataset(tempPair, 0.5)
        return train_info, valid_info, test_info
    else:
        pair_info = torch.load(os.path.join(path, "pair_info"))
        pair_info = shuffle_dataset(pair_info)
        train_pair_info, rest_pair_info = split_dataset(pair_info, 0.8)
        valid_info, test_info = split_dataset(rest_pair_info, 0.5)
        torch.save(train_pair_info, os.path.join(path, "train_info"))
        torch.save(rest_pair_info, os.path.join(path, "test_info"))
        return train_pair_info, valid_info, test_info
    
def fold_test_split(path, fold):
    pair_info = torch.load(os.path.join(path, "pair_info"))
    n = len(pair_info) // 5
    temp = pair_info[fold*n:(fold+1)*n]
    valid_set = temp[:n // 2]
    test_set = temp[n // 2:]
    return pair_info[0:fold*n] + pair_info[(fold+1)*n:], valid_set, test_set

def inference_distribute(train_info, dev_info, test_info, ratio, close_infer=False):
    def check(v):
#         if v < 10**-2.5 or v > 10**5:
#             return False
#         else:
#             return True
        return True
    def get_inference_info(x):
        info = []
        for i, item in enumerate(x):
            if len(item[-1]) > 1:
                info.append(i)
        return info
    def close(index, x):
        new_x = []
        for i, item in enumerate(x):
            if i not in index and check(item[2]):
                new_x.append(item)
        return new_x
    def trans(n, ind, x):
        if n <= 0:
            return [], x
        else:
            n = int(n)
            index = ind[:n]
            new_x = []
            not_x = []
            for i, item in enumerate(x):
                if check(item[2]):
                    if i not in index:
                        new_x.append(item)
                    else:
                        not_x.append(item)
            return not_x, new_x
    def trans_add(n, pool, x):
        if n >= 0:
            return pool, x
        else:
            n = int(n)
            x.extend(pool[:-n])
            return pool[-n:], x
        
    str_info = ''
    train_infer_info = get_inference_info(train_info)
    dev_infer_info = get_inference_info(dev_info)
    test_infer_info = get_inference_info(test_info)

    str_info += f'Using ratio: {ratio}, close_infer: {close_infer}\n'
    str_info += f'before re-distribute: {len(train_infer_info)}, {len(dev_infer_info)}, {len(test_infer_info)} infered samples in train, dev and test dataset respectively.\n'

    if close_infer:
        str_info += "closed infered\n"
        print(str_info)
        return close(train_infer_info, train_info), close(dev_infer_info, dev_info), close(test_infer_info, test_info)
    else:
        total = len(train_infer_info) + len(dev_infer_info) + len(test_infer_info)
        num = np.floor(np.array(ratio) * total) - np.array([len(train_infer_info), len(dev_infer_info), len(test_infer_info)])
        info_pool = []
        extra, train_info = trans(num[0], train_infer_info, train_info)
        info_pool.extend(extra)
        extra, dev_info = trans(num[1], dev_infer_info, dev_info)
        info_pool.extend(extra)
        extra, test_info = trans(num[2], test_infer_info, test_info)
        info_pool.extend(extra)
        
        info_pool, train_info = trans_add(num[0], info_pool, train_info)
        info_pool, dev_info = trans_add(num[1], info_pool, dev_info)
        info_pool, test_info = trans_add(num[2], info_pool, test_info)
        str_info += f'after re-distribute: {ratio[0]*total}, {ratio[1]*total}, {ratio[2]*total} infered samples in train, dev and test dataset respectively.\n'
        print(str_info)
        return train_info, dev_info, test_info

def getFoldPair(pair, fold):
    gap = len(pair) // 5
    return pair[:fold*gap]+pair[(fold+1)*gap:],pair[fold*gap:(fold+1)*gap]
    
def train(
    Model,
    ModelType,
    args,
    lr,
    radius,
    setting,
    loss_fn,
    optim,
    schedule,
    schedule_args,
    batchsize,
    Epoch,
    train_info,
    embeding_path,
    smiles_path,
    device,
    source_path,
    dataset_split_func,
    split_args,
    close_infer,
    molType,
    fold=None
):
    """Output files."""
    file_model = f'../Results/{ModelType}/{train_info}/model/'
    if not os.path.exists(file_model):
        os.makedirs(file_model)
    # 复制源代码
    shutil.copyfile(source_path, os.path.join(file_model, "model.py"))
    with open(os.path.join(file_model, "parameters.json"), 'w') as f:
        json.dump({'Model args':args, 'Schedule args':schedule_args, 'lr':lr, 'radius':radius, 'batchsize':batchsize}, f)
    file_model += setting
    """Train setting."""
    train_pair_info, valid_pair_info, test_pair_info = dataset_split_func(**split_args)
    train_pair_info, valid_pair_info, test_pair_info = inference_distribute(train_pair_info, valid_pair_info, test_pair_info, [0.8,0.1,0.1], close_infer)
    if fold is not None:
        train_pair_info, test_pair_info = getFoldPair(train_pair_info+valid_pair_info+test_pair_info, fold)
    if molType != 'Unimol':
        train_set = CustomDataSet(train_pair_info, embeding_path, smiles_path, args['mol_in_dim'], radius,Type=molType, log10=True)
        valid_set = CustomDataSet(test_pair_info, embeding_path, smiles_path, args['mol_in_dim'], radius, Type=molType, log10=True)
    else:
        train_set = ESMUniCustomDataset(train_pair_info, embeding_path, smiles_path, True)
        valid_set = ESMUniCustomDataset(test_pair_info, embeding_path, smiles_path, True)
    
    train_loader = CustomDataLoader(data=train_set, batch_size=batchsize, shuffle=True, drop_last=False, num_workers=50, prefetch_factor=20, persistent_workers=True, pin_memory=True)
    valid_loader = CustomDataLoader(data=valid_set, batch_size=batchsize, drop_last=False, num_workers=50, prefetch_factor=10, persistent_workers=True, pin_memory=True)
    model = Model(**args).to(device)
    optimizer = optim(model.parameters(), lr=lr)
    scheduler = schedule(optimizer, **schedule_args)
    tester = Tester(device, loss_fn)
    trainer = Trainer(device, loss_fn)
    print("start to training...")
    writer = SummaryWriter(f'../Results/{ModelType}/{train_info}/logs/')
    for epoch in range(1, Epoch + 1):
        train_MAE, train_rmse, train_r2, loss_train = trainer.run(model, train_loader, optimizer, len(train_pair_info), f"epoch {epoch} train:")
        MAE_dev, RMSE_dev, R2_dev, loss_dev = tester.test(model, valid_loader, len(valid_pair_info), desc=f"epoch {epoch} valid:")
        scheduler.step()
        writer.add_scalars("loss",{'train_loss':loss_train, 'dev_loss':loss_dev}, epoch)
        writer.add_scalars("RMSE",{'train_RMSE':train_rmse, 'dev_RMSE':RMSE_dev}, epoch)
        writer.add_scalars("MAE",{'train_MAE':train_MAE, 'dev_MAE':MAE_dev}, epoch)
        writer.add_scalars("R2",{'train_R2':train_r2, 'dev_R2':R2_dev}, epoch)
        tester.save_model(model, file_model+f'{molType}-trainR2:{train_r2:.4f}-devR2={R2_dev:.4f}-RMSE={RMSE_dev:.4f}-MAE={MAE_dev:.4f}-epoch={epoch}', R2_dev, epoch== Epoch)