"""
File: dataset.py
Author: Xiaowei Shen
Email: XiaoweiShen@buct.edu.cn
Created: 2023-11-29
Description: This file contains the implementation of datasets.

Copyright (c) 2023 Xiaowei Shen
This code is licensed under the MIT License.
See the LICENSE file for details.
"""
import os
import math
import torch
import numpy as np
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit import Chem, RDLogger
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader
RDLogger.DisableLog('rdApp.*')
fpgen = AllChem.GetRDKitFPGenerator(fpSize=1024)



def generateData(mol, proteins, value, nbits, radius, Type):
    if Type == "ECFP":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nbits).ToList()
    elif Type == "MACCSKeys":
        fp = MACCSkeys.GenMACCSKeys(mol).ToList()
    elif Type == "RDKIT":
        fp = fpgen.GetFingerprint(mol).ToList()
    data = Data(x = torch.FloatTensor(fp).unsqueeze(0), pro_emb=proteins, value=value)
    return data
    
class CustomDataSet(Dataset):
    def __init__(self, Pairinfo, ProteinsPath, SmilesIndexPath, nbits, radius, log10=False, Type='ECFP'):
        super(CustomDataSet, self).__init__()
        self.pairinfo = Pairinfo
        self.smiles = torch.load(SmilesIndexPath)
        self.seq_path = os.path.join(ProteinsPath, '{}.pt')
        self.nbits = nbits
        self.radius = radius
        self.log10 = log10
        self.Type = Type
        print(f"log10:{self.log10} molType:{self.Type}")
    def __getitem__(self, idx):
        pro_id = int(self.pairinfo[idx][0])
        smi_id = int(self.pairinfo[idx][1])
        value = self.pairinfo[idx][2]
        protein_emb = torch.load(self.seq_path.format(pro_id))
        mol = AllChem.MolFromSmiles(self.smiles[smi_id].strip())
        if self.log10:
            value = math.log10(value)
        else:
            value = math.log2(value)
        data = generateData(mol,  protein_emb, value, self.nbits, self.radius, self.Type)
        return data
    def collate_fn(self, batch):
        return Batch.from_data_list(batch, follow_batch=['pro_emb'])
    def __len__(self):
        return len(self.pairinfo)
    
def generate_ESMUni_data(mol, proteins, value):
    data = Data(x = mol, pro_emb=proteins, value=value)
    return data
    
class ESMUniCustomDataset(Dataset):
    def __init__(self, Pairinfo, ProteinsPath, SmilesPath, log10):
        super(ESMUniCustomDataset, self).__init__()
        self.seq_path = os.path.join(ProteinsPath, '{}.pt')
        self.mol_embeding = torch.load(SmilesPath)
        self.pairinfo = Pairinfo
        self.log10 = log10
        print(f"unimol Dataset, log10:{self.log10}")
    def __getitem__(self, idx):
        pro_id = int(self.pairinfo[idx][0])
        smi_id = int(self.pairinfo[idx][1])
        value = self.pairinfo[idx][2]
        protein_emb = torch.load(self.seq_path.format(pro_id))
        if self.log10:
            value = math.log10(value)
        else:
            value = math.log2(value)
        data = generate_ESMUni_data(self.mol_embeding[smi_id:smi_id+1, :],  protein_emb,  value)
        return data
    def collate_fn(self, batch):
        return Batch.from_data_list(batch, follow_batch=['pro_emb'])
    def __len__(self):
        return len(self.pairinfo)

class CustomDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)
        
def shuffle_dataset(dataset):
    np.random.shuffle(dataset)
    return dataset

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2