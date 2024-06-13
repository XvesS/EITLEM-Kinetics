## 相似性比较
## 获取相似性分数
import torch
# from Bio import SeqIO
import biotite.sequence as seq
import biotite.sequence.align as align
from tqdm import tqdm
import multiprocessing
import argparse
import torch
from rdkit import RDLogger, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
import re
import os
RDLogger.DisableLog('rdApp.*')

matrix = align.SubstitutionMatrix.std_protein_matrix()
def getScoreA(arg):
    j, searchSeq, seqIndex= arg
    q = seq.ProteinSequence(seqIndex[j])
    sim = []
    for i in searchSeq:
        alignments = align.align_optimal(q, seq.ProteinSequence(seqIndex[i]), matrix, local=False, gap_penalty=(-10, -0.5))
        t = align.get_sequence_identity(alignments[0])
        sim.append(t)
    torch.save(sim, f"./temp_seq_similarity/{j}")

def getScoreB(arg): 
    s, searchSeq = arg
    t = seq.ProteinSequence(re.sub(r'[UZOB]', 'X', s))
    sim = []
    for item in searchSeq:
        alignments = align.align_optimal(t, seq.ProteinSequence(re.sub(r'[UZOB]', 'X', item)), matrix, local=False, gap_penalty=(-10, -0.5))
        t = align.get_sequence_identity(alignments[0])
        sim.append(t)
    return (str(s), max(sim))


def getMolScore(arg):
    j, searchMol, molIndex = arg
    j_fp = MACCSkeys.GenMACCSKeys(AllChem.MolFromSmiles(molIndex[j]))
    sim = []
    for i in searchMol:
        i_fp = MACCSkeys.GenMACCSKeys(AllChem.MolFromSmiles(molIndex[i]))
        sim.append(DataStructs.FingerprintSimilarity(j_fp, i_fp))
    return (j, max(sim))

def getMolScoreB(arg):
    j, searchMol = arg
    j_fp = MACCSkeys.GenMACCSKeys(AllChem.MolFromSmiles(j))
    sim = []
    for i in searchMol:
        i_fp = MACCSkeys.GenMACCSKeys(AllChem.MolFromSmiles(i))
        sim.append(DataStructs.FingerprintSimilarity(j_fp, i_fp))
    return (j, max(sim))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--Type', type=int, required=False, default=1)
    parser.add_argument('-q', '--query', type=str, required=False)
    parser.add_argument('-s', '--search', type=str, required=False)
    parser.add_argument('-f', '--indexPath', type=str, required=False)
    parser.add_argument('-m', '--mol', type=bool, required=False, default=False)
    parser.add_argument('-r', '--resultPath', type=str, required=False, default=None)
    parser.add_argument('-l', '--loadType', type=str, required=False, default=None)
    return parser.parse_args()






def seq_smi(args):
    if args.Type == 0:
        if args.resultPath is None:
            raise ValueError
        if args.loadType is not None:
            queryPairinfo = torch.load(f"../Data/{args.loadType}/{args.loadType}TestPairInfo")
            searchPairInfo = torch.load(f"../Data/{args.loadType}/{args.loadType}TrainPairInfo")
            seqIndex = torch.load(f"../Data/Feature/index_seq")
        else:
            queryPairinfo = torch.load(args.query)
            searchPairInfo = torch.load(args.search)
            seqIndex = torch.load(args.indexPath)

        seqIndex = {k:re.sub(r'[UZOB]', 'X', v) for k, v in seqIndex.items()}

        # 设定进程池
        pool = multiprocessing.Pool()
        ## 设定工作目录
        os.makedirs("./temp_seq_similarity", exist_ok=True)
        # 获取训练集和测试集中的序列
        searchSeq = set()
        querySeq = set()
        [ searchSeq.add(item[0]) for item in searchPairInfo]
        [ querySeq.add(item[0]) for item in queryPairinfo]

        print(f"{len(querySeq)} query entrys \n {len(searchSeq)} search entrys")
        params = [ (i, searchSeq, seqIndex) for i in querySeq]
        pool.map(getScoreA, params)
        pool.close()
        pool.join()
        
        file_names = os.listdir("./temp_seq_similarity")
        result = {}
        for item in file_names:
            sim_data = torch.load(os.path.join("./temp_seq_similarity",item))
            result[int(item)] = sim_data
        torch.save(result, args.resultPath)

        os.system("rm -rf ./temp_seq_similarity")


    else:
        querySeq = set(torch.load(args.query))
        searchSeq = set(torch.load(args.search))
        pool = multiprocessing.Pool()
        result = { k:1.0 for k in (searchSeq & querySeq) }
        querySeq = querySeq - searchSeq
        print(f"{len(querySeq)} query entrys \n {len(searchSeq)} search entrys")
        params = [ (item, searchSeq) for item in  querySeq]
        res = pool.map(getScoreB, params)
        pool.close()
        pool.join()
        res = { k:v for k, v in res}
        result.update(res)
    torch.save(result, "./simiResults")


def mol_smi(args):
    if args.Type == 0:
        queryPairinfo = torch.load(args.query)
        searchPairInfo = torch.load(args.search)
        searchMol = set()
        queryMol = set()
        [ searchMol.add(item) for item in searchPairInfo]
        [ searchMol.add(item) for item in searchPairInfo]
        result = { k:1.0 for k in (searchMol & queryMol) }
        queryMol = queryMol - searchMol
        print(f"{len(queryMol)} query entrys \n {len(searchMol)} search entrys")
        params = [ (i, searchMol) for i in queryMol]
        res = pool.map(getMolScoreB, params)
        pool.close()
        pool.join()
        res = { k:v for k, v in res}
        result.update(res)
        torch.save(result, "./molSimiResults")
    else:
        queryPairinfo = torch.load(args.query)
        searchPairInfo = torch.load(args.search)
        pool = multiprocessing.Pool()
        index_smiles = torch.load(args.indexPath)
        searchMol = set()
        queryMol = set()
        [ searchMol.add(item[1]) for item in searchPairInfo]
        [ queryMol.add(item[1]) for item in queryPairinfo]
        result = { k:1.0 for k in (searchMol & queryMol) }
        queryMol = queryMol - searchMol
        print(f"{len(queryMol)} query entrys \n {len(searchMol)} search entrys")
        params = [ (i, searchMol, index_smiles) for i in queryMol]
        res = pool.map(getMolScore, params)
        pool.close()
        pool.join()
        res = { k:v for k, v in res}
        result.update(res)
        torch.save(result, "./molSimiResults")


def all_dataset_similarity(Type):
    query = torch.load(f"../Data/{Type}/{Type}TestPairInfo")
    search = torch.load(f"../Data/{Type}/{Type}TrainPairInfo")
    if Type != "KKM":
        search += torch.load("../Data/KKM/KKMTrainPairInfo")
    else:
        search += torch.load("../Data/KCAT/KCATTrainPairInfo")
        search += torch.load("../Data/KM/KMTrainPairInfo")
    seqIndex = torch.load("../Data/Feature/index_seq")

    seqIndex = {k:re.sub(r'[UZOB]', 'X', v) for k, v in seqIndex.items()}
    pool = multiprocessing.Pool()
    os.makedirs(f"./temp_seq_similarity", exist_ok=True)

    query = set([ item[0] for item in query])
    search = set([ item[0] for item in search])
    query = query - search
    query = list(query)
    search = list(search)
    print(f"{len(query)} query entrys \n {len(search)} search entrys")
    params = [ (i, search, seqIndex) for i in query]
    
    pool.map(getScoreA, params)
    pool.close()
    pool.join()
    
    file_names = os.listdir("./temp_seq_similarity")
    result = {}
    for item in file_names:
        sim_data = torch.load(os.path.join("./temp_seq_similarity",item))
        result[int(item)] = sim_data
    
    result['index'] = search

    torch.save(result, f"../Data/{Type}/all_seq_Indentity")
    os.system("rm -rf ./temp_seq_similarity")


if __name__ == '__main__':
    args = parse_args()
    if not args.mol:
        seq_smi(args)
    else:
        mol_smi(args)
    # all_dataset_similarity('KCAT')
    # all_dataset_similarity('KM')
    # all_dataset_similarity('KKM')
