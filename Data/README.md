# Data for EITLEM
This directory stores data for training and testing

### Training and Testing
The files for the training (Testing) set are saved in the folder corresponding to the kinetics parameter name. i.e. ./KCAT/KCATTrainPairInfo.  
(Note:Any file without a special suffix can be loaded using `torch.load`)  

`xxxTrainPairInfo` contains a `List` object like [ [7384, 6962, 0.132, [23918]],[12499, 5047, 23.86, [5176]] , ...], with each element representing a data sample. For example, in [7384, 6962, 0.132, [23918]], 7384 indicates the index of the protein sequence, 6962 indicates the index of the substrate SMILES representation, 0.132 represents the experimentally measured kinetic parameter value, and 23918 indicates the index order of this sample in the original dataset.

The index files stored in `./Feature/`, named `index_smiles` and `index_seq`.
you can use following code to get original data:
```python
import torch
import json
# open files
kcat_train_pair = torch.load("./KCAT/KCATTrainPairInfo")
index_smiles = torch.load("./Feature/index_smiles")
index_seqs = torch.load("./Feature/index_seq")
with open("./KCAT/kcat_data.json", 'r') as f:
    raw_data = json.load(f)

# get a sample
sample = kcat_train_pair[0]

# print
print(index_seqs[sample[0]]) # seqs
print(index_smiles[sample[1]]) # smiles for substrate
print(sample[2]) # experimental kcat value
print(raw_data[sample[3][0]]) # raw data
```


