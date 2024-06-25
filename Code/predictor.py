import sys
from dataset import EitlemDataSet, EitlemDataLoader
from KMP import EitlemKmPredictor
from ensemble import ensemble
from KCM import EitlemKcatPredictor
from KKMP import EitlemKKmPredictor
from tqdm import tqdm
import torch
 
def predict(Type, modelPath, pairInfo, embedingPath, smilesPath, log10, device, molType):
    Dataset = EitlemDataSet(pairInfo, embedingPath, smilesPath, 1024, 4, log10, molType)
    Loader = EitlemDataLoader(data=Dataset, batch_size=100, shuffle=False, drop_last=False, num_workers=50, prefetch_factor=10, persistent_workers=False, pin_memory=False)
    if molType == 'MACCSKeys':
        molDim = 167 
    else:
        molDim = 1024
    if Type == 'KCAT':
        model = EitlemKcatPredictor(molDim, 512, 1280, 10, 0.5, 10)
    elif Type == 'KM':
        model = EitlemKmPredictor(molDim, 512, 1280, 10, 0.5, 10)
    elif Type == 'KKM':
        model = ensemble(molDim, 512, 1280, 10, 0.5, 10)
    elif Type == 'KKMP':
        model = EitlemKKmPredictor(molDim, 512, 1280, 10, 0.5, 10)
        
    model.load_state_dict(torch.load(modelPath))
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        testY, testPredict = [], []
        for data in tqdm(Loader):
            pre = model(data.to(device))
            testY.extend(data.value.cpu().tolist())
            testPredict.extend(pre.cpu().tolist())
        return testY, testPredict