from torch import nn
import sys
import re
import torch
from eitlem_utils import Tester, Trainer, get_pair_info
from KCM import EitlemKcatPredictor
from KMP import EitlemKmPredictor
from ensemble import ensemble
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import EitlemDataSet, EitlemDataLoader
import os
import shutil
import argparse


def kineticsTrainer(kkmPath, TrainType, Type, Iteration, log10, molType, device):
    train_info = f"Transfer-{TrainType}-{Type}-train-{Iteration}"

    if os.path.exists(f'../Results/{Type}/{train_info}'):
        return None
    
    if kkmPath is not None:
        Epoch = 40 // (Iteration // 2)
    else:
        Epoch = 100
    
    file_model = f'../Results/{Type}/{train_info}/Weight/'
    
    if kkmPath is not None:
        trained_weights = torch.load(kkmPath)
        if Type == 'KCAT':
            if molType == 'MACCSKeys':
                model = EitlemKcatPredictor(167, 512, 1280, 10, 0.5, 10)
            else:
                model = EitlemKcatPredictor(1024, 512, 1280, 10, 0.5, 10)
            weights = model.state_dict()
            pretrained_para = {k[5:]: v for k, v in trained_weights.items() if 'kcat' in k and k[5:] in weights}
            weights.update(pretrained_para)
            model.load_state_dict(weights)
        else:
            if molType == 'MACCSKeys':
                model = EitlemKmPredictor(167, 512, 1280, 10, 0.5, 10)
            else:
                model = EitlemKmPredictor(1024, 512, 1280, 10, 0.5, 10)
            weights = model.state_dict()
            pretrained_para = {k[3:]: v for k, v in trained_weights.items() if 'km' in k and k[3:] in weights}
            weights.update(pretrained_para)
            model.load_state_dict(weights)
    else:
        if Type == 'KCAT':
            if molType == 'MACCSKeys':
                model = EitlemKcatPredictor(167, 512, 1280, 10, 0.5, 10)
            else:
                model = EitlemKcatPredictor(1024, 512, 1280, 10, 0.5, 10)
        else:
            if molType == 'MACCSKeys':
                model = EitlemKmPredictor(167, 512, 1280, 10, 0.5, 10)
            else:
                model = EitlemKmPredictor(1024, 512, 1280, 10, 0.5, 10)
    
    if not os.path.exists(file_model):
        os.makedirs(file_model)
    file_model += 'Eitlem_'
    """Train setting."""
    train_pair_info, test_pair_info = get_pair_info("../Data/", Type, False)
    train_set = EitlemDataSet(train_pair_info, f'../Data/Feature/esm1v_t33_650M_UR90S_1_embeding_1280/', f'../Data/Feature/index_smiles', 1024, 4, log10, molType)
    test_set = EitlemDataSet(test_pair_info, f'../Data/Feature/esm1v_t33_650M_UR90S_1_embeding_1280/', f'../Data/Feature/index_smiles', 1024, 4, log10, molType)
    train_loader = EitlemDataLoader(data=train_set, batch_size=200, shuffle=True, drop_last=False, num_workers=30, prefetch_factor=5, persistent_workers=True, pin_memory=True)
    valid_loader = EitlemDataLoader(data=test_set, batch_size=200, drop_last=False, num_workers=30, prefetch_factor=5, persistent_workers=True, pin_memory=True)
    model = model.to(device)
    if kkmPath is not None:
        out_param = list(map(id, model.out.parameters()))
        rest_param = filter(lambda x:id(x) not in out_param, model.parameters())
        optimizer = torch.optim.AdamW([
                                       {'params': rest_param, 'lr':1e-4},
                                       {'params':model.out.parameters(), 'lr':1e-3}, 
                                      ])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.8)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.9)
    
    loss_fn = nn.MSELoss()
    tester = Tester(device, loss_fn, log10=log10)
    trainer = Trainer(device, loss_fn, log10=log10)
    
    print("start to train...")
    writer = SummaryWriter(f'../Results/{Type}/{train_info}/logs/')
    for epoch in range(1, Epoch + 1):
        train_MAE, train_rmse, train_r2, loss_train = trainer.run(model, train_loader, optimizer, len(train_pair_info), f"{Iteration}iter epoch {epoch} train:")
        MAE_dev, RMSE_dev, R2_dev, loss_dev = tester.test(model, valid_loader, len(test_pair_info), desc=f"{Iteration}iter epoch {epoch} valid:")
        scheduler.step()
        writer.add_scalars("loss",{'train_loss':loss_train, 'dev_loss':loss_dev}, epoch)
        writer.add_scalars("RMSE",{'train_RMSE':train_rmse, 'dev_RMSE':RMSE_dev}, epoch)
        writer.add_scalars("MAE",{'train_MAE':train_MAE, 'dev_MAE':MAE_dev}, epoch)
        writer.add_scalars("R2",{'train_R2':train_r2, 'dev_R2':R2_dev}, epoch)
        tester.save_model(model, file_model+f'{molType}_trainR2_{train_r2:.4f}_devR2_{R2_dev:.4f}_RMSE_{RMSE_dev:.4f}_MAE_{MAE_dev:.4f}') # 保存
    pass


def KKMTrainer(kcatPath, kmPath, TrainType, Iteration, log10, molType, device):
    train_info = f"Transfer-{TrainType}-KKM-train-{Iteration}"
    if os.path.exists(f'../Results/KKM/{train_info}'):
        return None
    
    Epoch = 40
    file_model = f'../Results/KKM/{train_info}/Weight/'
    if molType == 'MACCSKeys':
        model = ensemble(167, 512, 1280, 10, 0.5, 10)
    else:
        model = ensemble(1024, 512, 1280, 10, 0.5, 10)
    kcat_pretrained = torch.load(kcatPath)
    km_pretrained = torch.load(kmPath)
    kcat_parameters = model.kcat.state_dict()
    km_parameters = model.km.state_dict()
    pretrained_kcat_para = {k:v for k, v in kcat_pretrained.items() if k in kcat_parameters}
    pretrained_km_para = {k:v for k, v in km_pretrained.items() if k in km_parameters}
    kcat_parameters.update(pretrained_kcat_para)
    km_parameters.update(pretrained_km_para)
    model.kcat.load_state_dict(kcat_parameters)
    model.km.load_state_dict(km_parameters)
    if not os.path.exists(file_model):
        os.makedirs(file_model)

    file_model += 'Eitlem_'
    """Train setting."""
    train_pair_info, test_pair_info = get_pair_info("../Data/", 'KKM')
    train_set = EitlemDataSet(train_pair_info, f'../Data/Feature/esm1v_t33_650M_UR90S_1_embeding_1280/', f'../Data/Feature/index_smiles', 1024, 4, log10, molType)
    test_set = EitlemDataSet(test_pair_info, f'../Data/Feature/esm1v_t33_650M_UR90S_1_embeding_1280/', f'../Data/Feature/index_smiles', 1024, 4, log10, molType)
    train_loader = EitlemDataLoader(data=train_set, batch_size=200, shuffle=True, drop_last=False, num_workers=60, prefetch_factor=10, persistent_workers=True, pin_memory=True)
    valid_loader = EitlemDataLoader(data=test_set, batch_size=200, drop_last=False, num_workers=60, prefetch_factor=10, persistent_workers=True, pin_memory=True)
    model = model.to(device)
    optimizer = torch.optim.AdamW([
                                   {'params': model.kcat.parameters(), 'lr':1e-4},
                                   {'params':model.km.parameters(), 'lr':1e-4},
                                   {'params':model.o.parameters(), 'lr':1e-3},
                                  ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.9)
    loss_fn = nn.MSELoss()
    tester = Tester(device, loss_fn, log10=log10)
    trainer = Trainer(device, loss_fn, log10=log10)
    print("start to train...")
    writer = SummaryWriter(f'../Results/KKM/{train_info}/logs/')
    for epoch in range(1, Epoch + 1):
        train_MAE, train_rmse, train_r2, loss_train = trainer.run(model, train_loader, optimizer, len(train_set), f"{Iteration}iter epoch {epoch} train:")
        MAE_dev, RMSE_dev, R2_dev, loss_dev = tester.test(model, valid_loader, len(test_set), desc=f"{Iteration}iter epoch {epoch} valid:")
        scheduler.step()
        writer.add_scalars("loss",{'train_loss':loss_train, 'dev_loss':loss_dev}, epoch)
        writer.add_scalars("RMSE",{'train_RMSE':train_rmse, 'dev_RMSE':RMSE_dev}, epoch)
        writer.add_scalars("MAE",{'train_MAE':train_MAE, 'dev_MAE':MAE_dev}, epoch)
        writer.add_scalars("R2",{'train_R2':train_r2, 'dev_R2':R2_dev}, epoch)
        tester.save_model(model, file_model+f'{molType}_trainR2:{train_r2:.4f}_devR2_{R2_dev:.4f}_RMSE_{RMSE_dev:.4f}_MAE_{MAE_dev:.4f}')


def getPath(Type, TrainType, Iteration):
    train_info = f"Transfer-{TrainType}-{Type}-train-{Iteration}"
    file_model = f'../Results/{Type}/{train_info}/Weight/'
    fileList = os.listdir(file_model)
    return os.path.join(file_model, fileList[0])

def TransferLearing(Iterations, TrainType, log10=False, molType='MACCSKeys', device=None):
    for iteration in range(1, Iterations + 1):
        if iteration == 1:
            kineticsTrainer(None, TrainType, 'KCAT', iteration, log10, molType, device)
            kineticsTrainer(None, TrainType, 'KM', iteration, log10, molType, device)
        else:
            kkmPath = getPath('KKM', TrainType, iteration-1)
            kineticsTrainer(kkmPath, TrainType, 'KCAT', iteration, log10, molType, device)
            kineticsTrainer(kkmPath, TrainType, 'KM', iteration, log10, molType, device)
        
        kcatPath = getPath('KCAT', TrainType, iteration)
        kmPath = getPath('KM', TrainType, iteration)
        KKMTrainer(kcatPath, kmPath, TrainType, iteration, log10, molType, device)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--Iteration', type=int, required=True)
    parser.add_argument('-t', '--TrainType', type=str, required=True)
    parser.add_argument('-l', '--log10', type=bool, required=False, default=True)
    parser.add_argument('-m', '--molType', type=str, required=False, default='MACCSKeys')
    parser.add_argument('-d', '--device', type=int, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')
    print(f"use device {device}")
    TransferLearing(args.Iteration, args.TrainType, args.log10, args.molType, device)
