"""
File: model.py
Author: Xiaowei Shen
Email: XiaoweiShen@buct.edu.cn
Created: 2023-11-29
Description: This file contains the implementation of models.

Copyright (c) 2023 Xiaowei Shen
This code is licensed under the MIT License.
See the LICENSE file for details.
"""
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_geometric.utils import softmax
from torch_geometric.nn import global_add_pool

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
        self.merge = nn.Linear(2*hidden_dim, 1, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, mol, prot, batch):
        q = F.relu(self.q(mol))
        r = q.repeat_interleave(degree(batch,  dtype=batch.dtype), dim=0)
        k = F.relu(self.k(prot))
        score = self.merge(torch.cat([k, r], dim=-1))
        score = softmax(score, batch, dim=0)
        o = global_add_pool(k * score, batch)
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

class Predictor(nn.Module):
    def __init__(self, 
                 mol_in_dim, 
                 hidden_dim=128, 
                 protein_dim=1280, 
                 layer=10, 
                 dropout=0.2, 
                 att_layer=10
                ):
        super(Predictor, self).__init__()
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

class ensemble(nn.Module):
    def __init__(self, mol_in_dim, hidden_dim, protein_dim, layer, dropout, att_layer):
        super().__init__()
        self.kcat = Predictor(mol_in_dim, hidden_dim, protein_dim, layer, dropout, att_layer)
        self.km = Predictor(mol_in_dim, hidden_dim, protein_dim, layer, dropout,att_layer)
        self.o = nn.Sequential(
            nn.Linear(2*hidden_dim, 4*hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(4*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, data):
        kcat_out = self.kcat(data.x, data.pro_emb, data.pro_emb_batch)
        km_out = self.km(data.x, data.pro_emb, data.pro_emb_batch)
        return self.o(torch.cat([kcat_out,km_out], dim=-1)).squeeze(dim=-1)