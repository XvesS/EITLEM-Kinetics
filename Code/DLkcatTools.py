import sys
# sys.path.append("../../KCAT/DLKcat/Code/model/")
# import model
import torch
import numpy as np
# 加载数据集
from Bio import SeqIO
from rdkit.Chem import AllChem
from rdkit import Chem
import math
from collections import defaultdict
from tqdm import tqdm
import pickle
from DLkcat_model import KcatPrediction


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

fingerprint_dict = load_pickle('../Data/DLkcat/fingerprint_dict.pickle')
atom_dict = load_pickle('../Data/DLkcat/atom_dict.pickle')
bond_dict = load_pickle('../Data/DLkcat/bond_dict.pickle')
edge_dict = load_pickle('../Data/DLkcat/edge_dict.pickle')
word_dict = load_pickle('../Data/DLkcat/sequence_dict.pickle')

n_fingerprint = len(fingerprint_dict)
n_word = len(word_dict)
n_edge = len(edge_dict)

radius=2
ngram=3
dim=10
layer_gnn=3
side=5
window=11
layer_cnn=3
layer_output=3
lr=1e-3
lr_decay=0.5
decay_interval=10
weight_decay=1e-6
device = torch.device('cuda:0')


def split_sequence(sequence, ngram):
    sequence = '-' + sequence + '='
    words = list()
    for i in range(len(sequence)-ngram+1) :
        try :
            words.append(word_dict[sequence[i:i+ngram]])
        except :
            word_dict[sequence[i:i+ngram]] = 0
            words.append(word_dict[sequence[i:i+ngram]])

    return np.array(words)

def create_atoms(mol):
    """Create a list of atom (e.g., hydrogen and oxygen) IDs
    considering the aromaticity."""
    # atom_dict = defaultdict(lambda: len(atom_dict))
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    # print(atoms)
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]

    return np.array(atoms)

def create_ijbonddict(mol):
    """Create a dictionary, which each key is a node ID
    and each value is the tuples of its neighboring node
    and bond (e.g., single and double) IDs."""
    # bond_dict = defaultdict(lambda: len(bond_dict))
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict

def extract_fingerprints(atoms, i_jbond_dict, radius):
    """Extract the r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using Weisfeiler-Lehman algorithm."""

    # fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    # edge_dict = defaultdict(lambda: len(edge_dict))

    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges
            (i.e., r-radius subgraphs or fingerprints)."""
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                # fingerprints.append(fingerprint_dict[fingerprint])
                # fingerprints.append(fingerprint_dict.get(fingerprint))
                try :
                    fingerprints.append(fingerprint_dict[fingerprint])
                except :
                    fingerprint_dict[fingerprint] = 0
                    fingerprints.append(fingerprint_dict[fingerprint])

            nodes = fingerprints

            """Also update each edge ID considering two nodes
            on its both sides."""
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    # edge = edge_dict[(both_side, edge)]
                    # edge = edge_dict.get((both_side, edge))
                    try :
                        edge = edge_dict[(both_side, edge)]
                    except :
                        edge_dict[(both_side, edge)] = 0
                        edge = edge_dict[(both_side, edge)]

                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints)

def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)

def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dict(dictionary), file)

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]

class Predictor(object):
    def __init__(self, model):
        self.model = model

    def predict(self, data):
        with torch.no_grad():
            predicted_value = self.model.forward(data)
        return predicted_value
    
def getDlkcatPredictor(model_path):
    Kcat_model = KcatPrediction(device, n_fingerprint, n_word, dim, layer_gnn, window, layer_cnn, layer_output).to(device)
    Kcat_model.load_state_dict(torch.load(model_path, map_location=device))
    Kcat_model.eval()
    predictor = Predictor(Kcat_model)
    return predictor

def dlkcatPredict(pairInfo, model_path):
    preValue = []
    tarValue = []
    seqIndex = torch.load("../Data/Feature/index_seq")
    smilesIndex = torch.load("../Data/Feature/index_smiles")
    predictor = getDlkcatPredictor(model_path)
    for pair in tqdm(pairInfo):
        smiles = smilesIndex[pair[1]]
        if '.' not in smiles:
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            try :
                atoms = create_atoms(mol)
                i_jbond_dict = create_ijbonddict(mol)
                fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius)
                adjacency = create_adjacency(mol)
                words = split_sequence(seqIndex[pair[0]],ngram)
                fingerprints = torch.LongTensor(fingerprints).to(device)
                adjacency = torch.FloatTensor(adjacency).to(device)
                words = torch.LongTensor(words).to(device)
                inputs = [fingerprints, adjacency, words]
                prediction = predictor.predict(inputs)
                Kcat_log_value = prediction.item()
                Kcat_value = math.pow(2,Kcat_log_value)
                preValue.append(Kcat_value)
                tarValue.append(pair[2])
            except:
                pass
    return np.log10(preValue), np.log10(tarValue)