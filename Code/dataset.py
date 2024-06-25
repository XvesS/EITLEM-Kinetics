import torch
import os
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import numpy as np
import math
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
### if an error occurs here, please check the version of rdkit 
fpgen = AllChem.GetRDKitFPGenerator(fpSize=1024)



def generateData(mol, proteins, value, nbits, radius, Type):
    if Type == "ECFP":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nbits).ToList()
    elif Type == "MACCSKeys":
        fp = MACCSkeys.GenMACCSKeys(mol).ToList()
    elif Type == "RDKIT":
        fp = fpgen.GetFingerprint(mol).ToList()
    elif Type == 'MACCSKeys_RDKIT':
        fp1 = MACCSkeys.GenMACCSKeys(mol).ToList()
        fp2 = fpgen.GetFingerprint(mol).ToList()
        return Data(x = torch.FloatTensor(fp1).unsqueeze(0), y=torch.FloatTensor(fp2).unsqueeze(0), pro_emb=proteins, value=value)
    data = Data(x = torch.FloatTensor(fp).unsqueeze(0), pro_emb=proteins, value=value)
    return data
    
class EitlemDataSet(Dataset):
    def __init__(self, Pairinfo, ProteinsPath, Smiles, nbits, radius, log10=False, Type='ECFP'):
        super(EitlemDataSet, self).__init__()
        self.pairinfo = Pairinfo
        if isinstance(Smiles, str):
            self.smiles = torch.load(Smiles)
        elif isinstance(Smiles, dict):
            self.smiles = Smiles
        self.seq_path = os.path.join(ProteinsPath, '{}.pt')
        self.nbits = nbits
        self.radius = radius
        self.log10 = log10
        self.Type = Type
        print(f"log10:{self.log10} molType:{self.Type}")
    def __getitem__(self, idx):
        pro_id = self.pairinfo[idx][0]
        smi_id = self.pairinfo[idx][1]
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
    
class EitlemDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)

def shuffle_dataset(dataset):
    np.random.shuffle(dataset)
    return dataset
def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2