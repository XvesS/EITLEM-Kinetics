import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import softmax
from torch_geometric.utils import degree
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
import math
from torch import autocast

class Resnet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)
        self.lin3 = nn.Linear(out_dim, out_dim)
        self.lin4 = nn.Linear(out_dim, out_dim)
    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = x + F.relu(self.lin2(x))
        x = x + F.relu(self.lin3(x))
        x = x + F.relu(self.lin4(x))
        return x
    
class ProMolAtt(nn.Module):
    def __init__(self, hidden_dim):
        super(ProMolAtt, self).__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.merge = nn.Linear(2*hidden_dim, 1, bias=False) # 计算相似性函数
        self.k = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, mol, prot, batch):
        q = F.relu(self.q(mol)) # 分子映射
        r = q.repeat_interleave(degree(batch,  dtype=batch.dtype), dim=0) # 分子扩增
        k = F.relu(self.k(prot))
        score = self.merge(torch.cat([k, r], dim=-1)) # 计算相似性分数
        score = softmax(score, batch, dim=0) # 权重加权
        o = global_add_pool(k * score, batch) # 聚合全局向量
        return o, q
    
class AttentionAgg(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionAgg, self).__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
    def forward(self, x, y):
        """
        x -> y ==> y^
        """
        q = F.relu(self.q(x.mean(dim=1)))
        k = F.relu(self.k(y))
        score = F.softmax(torch.matmul(q.unsqueeze(1), k.transpose(-1, -2)), dim=-1)
        out = torch.matmul(score, y).squeeze(1)
        return out
    
class MultiHeadAttenAgg(nn.Module):
    def __init__(self, hidden_dim, att_layer, dropout):
        super().__init__()
        self.seq_m = nn.ModuleList(AttentionAgg(hidden_dim) for _ in range(att_layer))
        self.seq_o = nn.Sequential(
        nn.Linear(hidden_dim*att_layer, 4*hidden_dim*att_layer),
        nn.ReLU(),
        nn.Dropout(p=dropout),
        nn.Linear(4*hidden_dim*att_layer, hidden_dim)
        )
    def forward(self, x, y):
        return self.seq_o(torch.cat([m(x, y) for m in self.seq_m], dim=-1))

class EitlemKmPredictor(nn.Module):
    def __init__(self, 
                 mol_in_dim, 
                 hidden_dim=128, 
                 protein_dim=1280, 
                 layer=10, 
                 dropout=0.2, 
                 att_layer=10
                ):
        super(EitlemKmPredictor, self).__init__()
        self.prej1 = Resnet(mol_in_dim, hidden_dim)
        self.prej2 = nn.Linear(protein_dim, hidden_dim, bias=False)
        self.pro_extrac = nn.ModuleList([ProMolAtt(hidden_dim) for _ in range(layer)])
        self.att1 = MultiHeadAttenAgg(hidden_dim,  att_layer, dropout)
        self.att2 = MultiHeadAttenAgg(hidden_dim,  att_layer, dropout)
        self.out = nn.Sequential(
            nn.Linear(2*hidden_dim, 4*hidden_dim),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(4*hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1)
        )
        print('F')
        
    def final_stage(self, mol, pro):
        pro_out = self.att1(mol, pro)
        mol_out = self.att2(pro, mol)
        return self.out(torch.cat([mol_out, pro_out], dim=-1)).squeeze(dim=-1)
    
    def forward(self, data):
        mol = F.relu(self.prej1(data.x))
        prot = F.relu(self.prej2(data.pro_emb))
        att_pro = []
        att_mol = []
        for m in self.pro_extrac:
            o, q = m(mol, prot, data.pro_emb_batch)
            att_pro.append(o)
            att_mol.append(q)
        att_mol = torch.stack(att_mol, dim=1)
        att_pro = torch.stack(att_pro, dim=1)
        return self.final_stage(att_mol, att_pro)

