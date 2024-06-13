#!/usr/bin/python
# coding: utf-8

# Author: LE YUAN
# Date: 2020-10-23

import pickle
import sys
import timeit
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_squared_error,r2_score
import json
import random
from tqdm import tqdm

class KcatPrediction(nn.Module):
    def __init__(self): 
        super(KcatPrediction, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(layer_gnn)])
        self.W_cnn = nn.ModuleList([nn.Conv2d(
                     in_channels=1, out_channels=1, kernel_size=2*window+1,
                     stride=1, padding=window) for _ in range(layer_cnn)])
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2*dim, 2*dim)
                                    for _ in range(layer_output)])
        # self.W_interaction = nn.Linear(2*dim, 2)
        self.W_interaction = nn.Linear(2*dim, 1)

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        # return torch.unsqueeze(torch.sum(xs, 0), 0)
        return torch.unsqueeze(torch.mean(xs, 0), 0)

    def attention_cnn(self, x, xs, layer):
        """The attention mechanism is applied to the last layer of CNN."""

        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs))
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)

        h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention(xs))
        weights = torch.tanh(F.linear(h, hs))
        ys = torch.t(weights) * hs

        # return torch.unsqueeze(torch.sum(ys, 0), 0)
        return torch.unsqueeze(torch.mean(ys, 0), 0)

    def forward(self, inputs):

        fingerprints, adjacency, words = inputs

        """Compound vector with GNN."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        compound_vector = self.gnn(fingerprint_vectors, adjacency, layer_gnn)

        """Protein vector with attention-CNN."""
        word_vectors = self.embed_word(words)
        protein_vector = self.attention_cnn(compound_vector,
                                            word_vectors, layer_cnn)

        """Concatenate the above two vectors and output the interaction."""
        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        for j in range(layer_output):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)
        # print(interaction)

        return interaction.squeeze(dim=-1)

    def __call__(self, data, train=True):

        inputs, correct_interaction = data[:-1], data[-1]
        predicted_interaction = self.forward(inputs)
        # print(predicted_interaction)

        if train:
            loss = F.mse_loss(predicted_interaction, correct_interaction)
            correct_values = correct_interaction.to('cpu').data.numpy()
            predicted_values = predicted_interaction.to('cpu').data.numpy()
            return loss, correct_values, predicted_values
        else:
            correct_values = correct_interaction.to('cpu').data.numpy()
            predicted_values = predicted_interaction.to('cpu').data.numpy()
            # correct_values = np.concatenate(correct_values)
            # predicted_values = np.concatenate(predicted_values)
            # ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            # predicted_values = list(map(lambda x: np.argmax(x), ys))
            # print(correct_values)
            # print(predicted_values)
            # predicted_scores = list(map(lambda x: x[1], ys))
            return correct_values, predicted_values


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        trainCorrect, trainPredict = [], []
        for data in tqdm(dataset, leave=False, desc="train:"):
            try:
                loss, correct_values, predicted_values = self.model(data)
            except:
                # print(idx, len(dataset))
                raise ValueError
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()

            correct_values = math.log10(math.pow(2,correct_values))
            predicted_values = math.log10(math.pow(2,predicted_values))
            trainCorrect.append(correct_values)
            trainPredict.append(predicted_values)
        rmse_train = np.sqrt(mean_squared_error(trainCorrect,trainPredict))
        r2_train = r2_score(trainCorrect,trainPredict)
        return loss_total, rmse_train, r2_train


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        N = len(dataset)
        SAE = 0  # sum absolute error.
        testY, testPredict = [], []
        for data in tqdm(dataset,leave=False, desc="train:") :
            (correct_values, predicted_values) = self.model(data, train=False)
            correct_values = math.log10(math.pow(2,correct_values))
            predicted_values = math.log10(math.pow(2,predicted_values))
            SAE += np.abs(predicted_values-correct_values)
            # SAE += sum(np.abs(predicted_values-correct_values))
            testY.append(correct_values)
            testPredict.append(predicted_values)
        MAE = SAE / N  # mean absolute error.
        rmse = np.sqrt(mean_squared_error(testY,testPredict))
        r2 = r2_score(testY,testPredict)
        return MAE, rmse, r2

    def save_MAEs(self, MAEs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, MAEs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in torch.load(file_name)]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

def seed_everything(seed):
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":

    """Hyperparameters."""
    radius = 2
    ngram  = 3
    dim = 10
    layer_gnn = 3 
    window = 11
    layer_cnn  =  3
    layer_output = 3 
    lr = 1e-3
    lr_decay = 0.5 
    decay_interval = 10 
    weight_decay = 1e-6
    iteration = 50
    setting = 'all--radius2--ngram3--dim20--layer_gnn3--window11--layer_cnn3--layer_output3--lr1e-3--lr_decay0.5--decay_interval10--weight_decay1e-6--iteration50'


    """CPU or GPU."""
    device = torch.device('cuda:1')

    """Load preprocessed data."""
    dir_input = ('../Data/DLkcat/')
    r_compounds = load_tensor(dir_input + 'compounds-0', torch.LongTensor)
    r_adjacencies = load_tensor(dir_input + 'adjacencies-0', torch.FloatTensor)
    r_proteins = load_tensor(dir_input + 'proteins-0', torch.LongTensor)
    r_interactions = load_tensor(dir_input + 'regression-0', torch.FloatTensor)
    
    t_compounds = load_tensor(dir_input + 'compounds-1', torch.LongTensor)
    t_adjacencies = load_tensor(dir_input + 'adjacencies-1', torch.FloatTensor)
    t_proteins = load_tensor(dir_input + 'proteins-1', torch.LongTensor)
    t_interactions = load_tensor(dir_input + 'regression-1', torch.FloatTensor)
    
    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
    word_dict = load_pickle(dir_input + 'sequence_dict.pickle')
    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)

    """Create a dataset and split it into train/dev/test."""
    test_dataset = list(zip(t_compounds, t_adjacencies, t_proteins, t_interactions))
    train_dataset = list(zip(r_compounds, r_adjacencies, r_proteins, r_interactions))
    
    # dev_dataset, test_dataset = split_dataset(test_dataset, 0.5)

    """Set a model."""
    seed_everything(1234)
    model = KcatPrediction().to(device)
    trainer = Trainer(model)
    tester = Tester(model)

    """Output files."""
    file_MAEs = '../Weights/DLkcat/MAEs--' + setting + '.txt'
    file_model = '..//Weights/DLkcat/' + setting
    MAEs = ('Epoch\tTime(sec)\tRMSE_train\tR2_train\tMAE_test\tRMSE_test\tR2_test')
    with open(file_MAEs, 'w') as f:
        f.write(MAEs + '\n')

    """Start training."""
    print('Training...')
    print(MAEs)
    start = timeit.default_timer()

    for epoch in range(1, iteration+1):

        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train, rmse_train, r2_train = trainer.train(train_dataset)
        # MAE_dev, RMSE_dev, R2_dev = tester.test(dev_dataset)
        MAE_test, RMSE_test, R2_test = tester.test(test_dataset)

        end = timeit.default_timer()
        time = end - start

        MAEs = [epoch, time, rmse_train, r2_train,
                MAE_test, RMSE_test, R2_test]
        tester.save_MAEs(MAEs, file_MAEs)
        tester.save_model(model, file_model)

        print('\t'.join(map(str, MAEs)))
