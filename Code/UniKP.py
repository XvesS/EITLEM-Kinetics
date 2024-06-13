import torch
from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq
import json
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc
import numpy as np
import random
import pickle
import math
from tqdm import trange
from sklearn.ensemble import ExtraTreesRegressor
from utils import split
from tools import metric
import os
from collections import defaultdict

def smiles_to_vec(Smiles):
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3
    mask_index = 4
    vocab = WordVocab.load_vocab('../Weights/UniKP/vocab.pkl')
    def get_inputs(sm):
        seq_len = 220
        sm = sm.split()
        if len(sm)>218:
            sm = sm[:109]+sm[-109:]
        ids = [vocab.stoi.get(token, unk_index) for token in sm]
        ids = [sos_index] + ids + [eos_index]
        seg = [1]*len(ids)
        padding = [pad_index]*(seq_len - len(ids))
        ids.extend(padding), seg.extend(padding)
        return ids, seg
    def get_array(smiles):
        x_id, x_seg = [], []
        for sm in smiles:
            a,b = get_inputs(sm)
            x_id.append(a)
            x_seg.append(b)
        return torch.tensor(x_id), torch.tensor(x_seg)
    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    trfm.load_state_dict(torch.load('../Weights/UniKP/trfm_12_23000.pkl'))
    trfm.eval()
    x_split = [split(sm) for sm in Smiles]
    xid, xseg = get_array(x_split)
    X = trfm.encode(torch.t(xid))
    return X


def Seq_to_vec(Sequence):
    for i in range(len(Sequence)):
        if len(Sequence[i]) > 1000:
            Sequence[i] = Sequence[i][:500] + Sequence[i][-500:]
    sequences_Example = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in Sequence]
    tokenizer = T5Tokenizer.from_pretrained("../Weights/UniKP/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("../Weights/UniKP/prot_t5_xl_uniref50")
    gc.collect()
    print(torch.cuda.is_available())
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    features = []
    batchsize = 100
    n_iter = len(sequences_Example) // batchsize
    for i in trange(n_iter+1, desc="Extract seq embeding"):
        sequence_batch = sequences_Example[i*batchsize:(i+1)*batchsize]
        if len(sequence_batch) > 0:
            ids = tokenizer.batch_encode_plus(sequence_batch, add_special_tokens=True, padding=True)
            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)
            with torch.no_grad():
                embedding = model(input_ids=input_ids, attention_mask=attention_mask)
            embedding = embedding.last_hidden_state.cpu().numpy()
            seq_len = (attention_mask == 1).sum(dim=-1)
            for item, l in zip(embedding, seq_len):
                features.append(item[:l - 1].mean(axis=0))
    return np.stack(features, axis=0)



class UniKp_predictor():
    def __init__(self):
        pass
    
    def _load_dataset(self, pair_info, index_seq, index_smiles):
        seqs = []
        smiles = []
        labels = []
        for item in pair_info:
            seqs.append(index_seq[item[0]])
            smiles.append(index_smiles[item[1]])
            labels.append(math.log10(item[2]))

        feature_mapping = np.array([ self.seq_mapping[item] for item in seqs])
        seqs_feature = self.seqs_feature[feature_mapping]
        smiles_feature = smiles_to_vec(smiles)
        feature = np.concatenate((smiles_feature, seqs_feature), axis=1)
        return feature, np.array(labels)
    

    def fit(self, train_pair_info, all_pair, index_seq, index_smiles, Type):
        self.index_seq = index_seq
        self.index_smiles = index_smiles
        self.Type = Type
        if os.path.exists(f"../Weights/UniKP/{Type}_prot_t5_embeding.pt"):
            embeding_info = torch.load(f"../Weights/UniKP/{Type}_prot_t5_embeding.pt")
            self.seqs_feature = embeding_info['embeding']
            self.seq_mapping = embeding_info['mapping']
        else:
            seqs = []
            for item in all_pair:
                seqs.append(self.index_seq[item[0]])
            seq_set = list(set(seqs))
            seq_set.sort(key=lambda x:len(x), reverse=True)
            self.seq_mapping = { v:k for k, v in enumerate(seq_set)}
            self.seqs_feature = Seq_to_vec(seq_set)
            torch.save({'embeding':self.seqs_feature, 'mapping':self.seq_mapping}, f"../Weights/UniKP/{self.Type}_prot_t5_embeding.pt")

        x_train, y_train = self._load_dataset(train_pair_info, self.index_seq, self.index_smiles)
        self.model = ExtraTreesRegressor(n_jobs=128)
        self.model.fit(x_train, y_train)
        
    
    def test(self, test_pair, ret=False):  
        x_test, y_test = self._load_dataset(test_pair, self.index_seq, self.index_smiles)
        y_pred = self.model.predict(x_test)
        if not ret:
            return metric(y_pred, y_test, True)
        else:
            return metric(y_pred, y_test, True), (y_pred, y_test)