import torch
import pandas as pd
from rdkit.Chem import AllChem,Descriptors,MACCSkeys
from rdkit import Chem
from rdkit.Chem import Draw
import re
import math
from rdkit.ML.Descriptors import MoleculeDescriptors
from collections import Counter
from torch_geometric.data import Data, Batch
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score,average_precision_score


def smile2v(m):
    try:
        residu = {}
        aa_smart = Chem.MolFromSmarts("[C:0](=[O:1])[C:2][N:3]")
        aa_num = len(m.GetSubstructMatches(aa_smart))
        chain = (aa_num-1)*"NCC(=O)"+"NCC=O"
#         print(chain)
        tmp=Chem.ReplaceCore(m,Chem.MolFromSmiles(chain),labelByIndex=True)
        if tmp:
            sub_list = Chem.MolToSmiles(tmp).split(".")
        else:
            aa_num = 10
            chain = (aa_num-1)*"NCC(=O)"+"NCC=O"
            tmp = Chem.ReplaceCore(m, Chem.MolFromSmiles(chain), labelByIndex=True)
            sub_list = Chem.MolToSmiles(tmp).split(".")

        for s in sub_list:
            index_list = re.findall( '\[(\d+)\*\]|\*', s)
            if "" in index_list:
                residu[str(["0" if y=="" else y for y in index_list])] = s
            else:
                residu[str(index_list)] = s
        return aa_num, residu
    except:
        return aa_num, None

def generate_adjacency_matrix(residue_list,n):
    if residue_list:
        for i in residue_list:
            if "," in i:
                index_list = i.strip("[]").split(",")
        connect_point = []
        for a in index_list:
            pep_num = int(a.strip(' \''))//4+1
            connect_point.append(pep_num)
        adjacency_matrix = [[x,x+1] for x in range(0,n,1)]
        adjacency_matrix.append(connect_point)
        adjacency_matrix = torch.tensor(adjacency_matrix).t().long()
        return adjacency_matrix
    else:
        return None

def generate_adjacency_matrix_ali(residue_list, miss_index):
    if residue_list:
        index_list = []
        for i in residue_list:
            if "," in i:
                index_list.append(i.strip("[]").split(","))
        connect_point = []
        for index_pair in index_list:
            temp_pair = []
            for index in index_pair:
                pep_num = int(index.strip(' \''))//4+1
                temp_pair.append(pep_num)
            connect_point.append(temp_pair)

        index_num_list = list(x for x in range(0,12,1))
        for  miss_i in miss_index:
            index_num_list.remove(miss_i)
        adjacency_matrix = [[index_num_list[x],index_num_list[x+1]] for x in range(len(index_num_list)-1)]
        for index_pair in connect_point:
            adjacency_matrix.append([index_num_list[index_pair[0]],index_num_list[index_pair[1]]])
        adjacency_matrix = torch.tensor(adjacency_matrix).t().long()
        return adjacency_matrix
    else:
        return None


def calc_chiral(mol):
    chiral_list = []
    for i, atom in enumerate(mol.GetAtoms()):
        chiral_list.append(str(atom.GetChiralTag()))
    return Counter(chiral_list)


def smi_to_descriptors_chiral(smile):
    mol = Chem.MolFromSmiles(smile)
    chiral_name = ["CHI_OTHER", "CHI_TETRAHEDRAL_CCW", "CHI_TETRAHEDRAL_CW", "CHI_UNSPECIFIED"]
    des_name = [x[0] for x in Descriptors._descList]

    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(des_name)
    if mol:
        descriptors = list(calculator.CalcDescriptors(mol))
        chiral_dict = calc_chiral(mol)
        for k in chiral_name:
            descriptors.append(chiral_dict[k])
    else:
        print(smile)

    return descriptors


def smi_to_descriptors(smile, feature_name):

    if feature_name == "Des":
        feature = smi_to_descriptors_chiral(smile)
    elif feature_name =="Morgan":
        feature = morganFP(smile)
    elif feature_name == "Maccs":
        feature = MACCSFingerPrint(smile)
    elif feature_name == "Topo":
        feature = TopoFingerPrint(smile)

    return feature


def morganFP(smile):
    mol= Chem.MolFromSmiles(smile)
    fingerprint = [x for x in AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
    return fingerprint

def MACCSFingerPrint(smile):
    mol = Chem.MolFromSmiles(smile)
    fingerprint = [x for x in MACCSkeys.GenMACCSKeys(mol)]
    return fingerprint

def TopoFingerPrint(smile):
    mol = Chem.MolFromSmiles(smile)
    fingerprint = [x for x in Chem.RDKFingerprint(mol)]
    return fingerprint


def node_feature(residue, feature_name, aa_num):
    node_feature_dic = {}
    node_feature_list = []
    feature_dim = {"Des":213, "Morgan":2048, "Maccs":167,"Topo":2048}
    for k, v in residue.items():
        #     print("key = "+k)
        if "," in k:
            index_list = k.strip("[]").split(",")
            index = min([int(x.strip(' \'')) for x in index_list])
        else:
            index = int(k.strip(' \'\[\]'))
        if index != 0:
            index = index // 4 + 1
            node_feature_dic[index] = smi_to_descriptors(re.sub('\[(\d+)\*\]|\*', "", v), feature_name)
        else:
            node_feature_dic[index] = smi_to_descriptors(v.strip("*"),feature_name)
    #     print(node_feature_dic.keys())
    # for k in range(max(node_feature_dic.keys()) + 1):
    for k in range(aa_num + 1):
        if k not in node_feature_dic.keys():

            node_feature_list.append(np.zeros(feature_dim[feature_name]))
        else:
            node_feature_list.append(node_feature_dic[k])
    return node_feature_list


def node_feature_count(residue):
    node_feature_index = []
    for k, v in residue.items():
        #     print("key = "+k)
        if "," in k:
            index_list = k.strip("[]").split(",")
            index = min([int(x.strip(' \'')) for x in index_list])
        else:
            index = int(k.strip(' \'\[\]'))
        if index != 0:
            index = index // 4 + 1
        node_feature_index.append(index)

    return node_feature_index


def Smile2Graph(smile, y, feature_name):
    mol = Chem.MolFromSmiles(smile)
    aa_num, residue = smile2v(mol)
    if residue:
        adjacency_matrix = generate_adjacency_matrix(residue, aa_num)
        feature = node_feature(residue,feature_name,aa_num)
        edge_index = torch.tensor(adjacency_matrix, dtype=torch.long)
        x = torch.tensor(feature, dtype=torch.float)
        x = torch.nan_to_num(x)
        y = torch.LongTensor([y])
        data = Data(x=x, edge_index=edge_index, y=y)
        return data
    else:
        return None

def Smile2Graph_global(smile, y, feature_name):
    mol = Chem.MolFromSmiles(smile)
    aa_num, residue = smile2v(mol)
    if residue:
        adjacency_matrix = generate_adjacency_matrix(residue, aa_num)
        feature = node_feature(residue,feature_name,aa_num)
        edge_index = torch.tensor(adjacency_matrix, dtype=torch.long)
        global_feature = smi_to_descriptors(smile,feature_name)
        global_feature = torch.tensor(global_feature,dtype=torch.float)
        x = torch.tensor(feature, dtype=torch.float)
        x = torch.nan_to_num(x)
        y = torch.LongTensor([y])
        data = MyData(x=x, edge_index=edge_index, y=y, global_feature = global_feature)
        return data
    else:
        return None

def Smile2Graph_global_ali(smile, y, feature_name):
    mol = Chem.MolFromSmiles(smile)
    aa_num, residue = smile2v(mol)
    if residue:
        adjacency_matrix = generate_adjacency_matrix(residue, aa_num)
        feature = node_feature(residue,feature_name,aa_num)
        edge_index = torch.tensor(adjacency_matrix, dtype=torch.long)
        global_feature = smi_to_descriptors(smile,feature_name)
        global_feature = torch.tensor(global_feature,dtype=torch.float)
        x = torch.tensor(feature, dtype=torch.float)
        x = torch.nan_to_num(x)
        y = torch.LongTensor([y])
        seq = node_feature_ali(residue,aa_num)
        miss_index = needleman_wunsch(seq)
        miss_index = torch.tensor(miss_index, dtype=torch.int8)
        data = MyData_miss(x=x, edge_index=edge_index, y=y, global_feature = global_feature,miss_index = miss_index)
        return data
    else:
        return None


# aligment before training

def Smile2Graph_global_ali_new(smile, y, feature_name):
    feature_dim = {"Des": 213, "Morgan": 2048, "Maccs": 167, "Topo":2048}
    mol = Chem.MolFromSmiles(smile)
    aa_num, residue = smile2v(mol)
    if residue:
        seq = node_feature_ali(residue,aa_num)
        miss_index = needleman_wunsch(seq)
        miss_index = torch.tensor(miss_index, dtype=torch.int8)

        adjacency_matrix = generate_adjacency_matrix_ali(residue,miss_index)
        feature = node_feature(residue,feature_name,aa_num)

        for i in miss_index:
            padding = [0 for i in range(feature_dim[feature_name])]
            feature.insert(i, padding)

        edge_index = torch.tensor(adjacency_matrix, dtype=torch.long)
        global_feature = smi_to_descriptors(smile,feature_name)
        global_feature = torch.tensor(global_feature,dtype=torch.float)
        x = torch.tensor(feature, dtype=torch.float)
        x = torch.nan_to_num(x)
        y = torch.LongTensor([y])

        data = MyData_miss(x=x, edge_index=edge_index, y=y, global_feature = global_feature,miss_index = miss_index)
        return data
    else:
        return None

class MyData_miss(Data):
    def __init__(self, x, edge_index, y, global_feature,miss_index):
        super(MyData_miss, self).__init__(x=x, edge_index=edge_index, y=y)
        self.global_feature = global_feature
        self.miss_index = miss_index


class MyData(Data):
    def __init__(self, x, edge_index, y, global_feature):
        super(MyData, self).__init__(x=x, edge_index=edge_index, y=y)
        self.global_feature = global_feature


def residu_counter(smile):
    mol = Chem.MolFromSmiles(smile)
    aa_num, residue = smile2v(mol)
    if residue:
        feature = node_feature_count(residue)
        return feature
    else:
        return None


def valiadation(y_test,y_score, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    acc = (tp+tn)/(tp+fp+tn+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    specificity = tn/(tn+fp)
    f1_score = 2*(recall*precision)/(recall+precision)
    mcc = (tp*tn-fp*fn)/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    AUROC = roc_auc_score(y_test,y_score)
    AUPRC = average_precision_score(y_test, y_score)
    performance = [recall,specificity,precision,acc,mcc,f1_score,AUROC,AUPRC]
    return performance


def needleman_wunsch(seq2, match_score=1, mismatch_score=-1, gap_penalty=-2):

    seq1 = ['-', 'CCN', '[C@@H](C)O', 'CCN', 'CCN', 'CCN', 'Cc1ccccc1', 'CC(C)C', 'CCN', 'CCN',
            '[C@@H](C)O']
    temp_faty = seq2.pop(0)
    m = len(seq1)
    n = len(seq2)
    score_matrix = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        score_matrix[i][0] = i * gap_penalty
    for j in range(n + 1):
        score_matrix[0][j] = j * gap_penalty

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = score_matrix[i - 1][j - 1] + (match_score if seq1[i - 1] == seq2[j - 1] else mismatch_score)
            delete = score_matrix[i - 1][j] + gap_penalty
            insert = score_matrix[i][j - 1] + gap_penalty
            score_matrix[i][j] = max(match, delete, insert)


    aligned_seq1 = []
    aligned_seq2 = []
    i = m
    j = n

    while i > 0 and j > 0:
        score = score_matrix[i][j]
        score_diag = score_matrix[i - 1][j - 1]
        score_up = score_matrix[i][j - 1]
        score_left = score_matrix[i - 1][j]

        if score == score_diag + (match_score if seq1[i - 1] == seq2[j - 1] else mismatch_score):
            aligned_seq1.insert(0, seq1[i - 1])
            aligned_seq2.insert(0, seq2[j - 1])
            i -= 1
            j -= 1
        elif score == score_left + gap_penalty:
            aligned_seq1.insert(0, seq1[i - 1])
            aligned_seq2.insert(0, "-")
            i -= 1
        else:
            aligned_seq1.insert(0, "-")
            aligned_seq2.insert(0, seq2[j - 1])
            j -= 1

    while i > 0:
        aligned_seq1.insert(0, seq1[i - 1])
        aligned_seq2.insert(0, "-")
        i -= 1

    while j > 0:
        aligned_seq1.insert(0, "-")
        aligned_seq2.insert(0, seq2[j - 1])
        j -= 1

    seq2_len = len(aligned_seq2)
    if seq2_len <= len(aligned_seq1):
        miss_index = [i+1 for i, x in enumerate(aligned_seq2) if x == '-']

    return miss_index


def node_feature_ali(residue,aa_num):
    node_feature_dic = {}
    node_feature_list = []
    circle_index_list = []
    for k, v in residue.items():

        if "," in k:
            index_list = k.strip("[]").split(",")
            index = min([int(x.strip(' \'')) for x in index_list])
            circle_site = True
            if index != 0:
                circle_index_list.append([index // 4 + 1, max([int(x.strip(' \'')) for x in index_list]) // 4 + 1])
            else:
                circle_index_list.append([0, max([int(x.strip(' \'')) for x in index_list]) // 4 + 1])
        else:
            index = int(k.strip(' \'\[\]'))
        if index != 0:
            index = index // 4 + 1
            node_feature_dic[index] = re.sub('\[(\d+)\*\]|\*', "", v)
        else:
            node_feature_dic[index] = v.strip("*")

    # for k in range(max(node_feature_dic.keys()) + 1):
    for k in range(aa_num + 1):
        if k not in node_feature_dic.keys():

            node_feature_list.append("H")
        else:
            node_feature_list.append(node_feature_dic[k])
    return node_feature_list
