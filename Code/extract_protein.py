import os
import torch
from tqdm import tqdm
cmd = 'python ./extract.py esm2_t33_650M_UR50D ../Data/Feature/seq_str.fasta ../Data/Feature/esm2_t33_650M_UR50D/ --repr_layers 33 --include per_tok'
os.system(cmd)
base = "../Data/Feature/esm2_t33_650M_UR50D/"
def change(index, layer):
    data = torch.load(base+f'{index}.pt')
    data = data['representations'][layer]
    torch.save(data, base+f'{index}.pt')
file_list = os.listdir(base)
length = len(file_list)
for index in tqdm(range(length)):
    change(index, 33)