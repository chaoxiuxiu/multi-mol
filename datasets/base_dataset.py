from argparse import Namespace
import json
import os.path
import random
import re
# from xeger import Xeger
import pandas as pd
import torch
import torchvision.transforms as transforms
from rdkit import Chem
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import numpy as np
import pyarrow as pa

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import io
from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.data.utils import get_class_sizes, get_data_from_smiles, get_task_names, split_data, get_data
from chemprop.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint, \
    makedirs, save_checkpoint
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from pretrain3d.utils.graph import smiles2graphwithface
from pretrain3d.utils.gt import isomorphic_core
from copy import deepcopy
import ast


class BaseDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 image_size: int,
                 names: list,
                 transforms,
                 smiles_column_name: str = "Smiles",
                 fingerprint_column_name: str = "Fingerprint",
                 k_100_column_name: str = 'k_100',
                 k_1000_column_name: str = 'k_1000',
                 k_10000_column_name: str = 'k_10000',
                 # mol_column_name: str = 'mol',  # New
                 pos_column_name: str = 'pos',  # New
                 label_name=None,
                 split='train',
                 is_pretrain=True,
                 MoleculeNet=True,
                 ocsr=False
                 ):
        self.image_size = image_size
        self.smiles_column_name = smiles_column_name
        self.names = names
        self.data_dir = data_dir
        self.split = split
        self.transforms = transforms
        self.is_pretrain = is_pretrain
        self.regex = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"

        if len(names) != 0:
            print(f"{data_dir}/{names[0]}.arrow")
            tables = [
                pa.ipc.RecordBatchFileReader(
                    pa.memory_map(f"{data_dir}/{name}.arrow", "r")
                ).read_all()
                for name in names
                if os.path.isfile(f"{data_dir}/{name}.arrow")
            ]

            self.table = pa.concat_tables(tables, promote=True)
            print(self.table.num_rows)

            if self.is_pretrain and smiles_column_name != "" and k_100_column_name != "":
                self.smiles_column_name = smiles_column_name
                self.fingerprint_column_name = fingerprint_column_name
                self.k_100_column_name = k_100_column_name
                # self.mol_column_name = mol_column_name  # New
                self.pos_column_name = pos_column_name  # New
                self.all_smiles = self.table[smiles_column_name].to_pandas().tolist()
                self.all_k_100 = self.table[k_100_column_name].to_pandas().tolist()
                self.all_k_1000 = self.table[k_1000_column_name].to_pandas().tolist()
                self.all_k_10000 = self.table[k_10000_column_name].to_pandas().tolist()
                # self.all_mol = self.table[mol_column_name].to_pandas().tolist()  # New
                self.all_pos = self.table[pos_column_name].to_pandas().tolist()  # New

            elif not self.is_pretrain and smiles_column_name != "" and MoleculeNet == True:
                self.smiles_column_name = smiles_column_name
                self.all_smiles = self.table[smiles_column_name].to_pandas().tolist()
                self.labels = self.table[label_name].to_pandas().tolist()
                self.all_pos = self.table[pos_column_name].to_pandas().tolist()

            elif not self.is_pretrain and smiles_column_name != "" and ocsr == True:
                self.smiles_column_name = smiles_column_name
                self.all_smiles = self.table[smiles_column_name].to_pandas().tolist()
                self.all_image_id = self.table['image_id'].to_pandas().tolist()
            else:
                self.all_smiles = list()
                self.all_fingerprint = list()
                self.all_k_100 = list()
                self.all_k_1000 = list()
                self.all_k_10000 = list()
                # self.all_mol = list()  # New
                self.all_pos = list()  # New
        else:
            self.all_smiles = list()
            self.all_fingerprint = list()
            self.all_k_100 = list()
            self.all_k_1000 = list()
            self.all_k_10000 = list()
            # self.all_mol = list()  # New
            self.all_pos = list()  # New

    def get_raw_image(self, index, image_key="images"):
        image_bytes = io.BytesIO(self.table[image_key][index].as_py())
        image_bytes.seek(0)
        return Image.open(image_bytes).convert("RGB")

    def get_image(self, index, image_key="images"):
        image = self.get_raw_image(index, image_key=image_key)
        image_tensor = self.transforms(image)
        return image_tensor

    def get_false_image(self, image_key="images"):
        random_index = random.randint(0, len(self.all_smiles) - 1)
        image = self.get_raw_image(random_index, image_key=image_key)
        image_tensor = self.transforms(image)
        return image_tensor

    def get_smiles(self, index):  # 微调
        smiles = self.all_smiles[index][0]  # mol
        return smiles

    def get_smile(self, index):  # pretrain
        smiles = self.all_smiles[index]  # mol
        return smiles

    def get_false_smiles(self, index):
        raw_smiles = self.all_smiles[index]
        smiles_false = self.change_smiles(raw_smiles)
        return smiles_false

    def get_k_100(self, index):
        k_100 = self.all_k_100[index]
        return k_100

    def get_k_1000(self, index):
        k_1000 = self.all_k_1000[index]
        return k_1000

    def get_k_10000(self, index):
        k_1000 = self.all_k_10000[index]
        return k_1000

    def get_label(self, index):
        assert not self.is_pretrain, "now, pretraining!! Just download stream tasks have labels."
        #return int(self.labels[index])
        return self.labels[index]

    def get_labels(self):
        labels = np.array(self.labels)
        return self.labels

    def get_oscr_smiles(self, index):
        smiles = self.all_smiles[index]
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol, doRandom=True)
        return smiles

    def get_pos(self, index):  # New
        # print(self.all_pos[index])
        #print(type(self.all_pos[index]))
        return self.all_pos[index]


    def get_data(self, index):
        mol = self.all_mol[index]
        pos = self.all_pos[index]
        pos_list_of_lists = ast.literal_eval(pos)
        pos = np.array(pos_list_of_lists)
        data = process_data(mol, pos, smiles2graphwithface, isomorphic_core)
        # print(f'data:',data)
        # print(f'pos:',pos)
        return data

    def __len__(self):
        return len(self.all_smiles)

    def getItem(self, index):
        image = self.get_image(index)
        image_false = self.get_false_image()
        # smiles = self.get_smiles(index)
        smiles = self.get_smile(index)
        # print(smiles)
        k_100 = self.get_k_100(index)
        k_1000 = self.get_k_1000(index)
        k_10000 = self.get_k_10000(index)
        smiles_false = self.get_false_smiles(index)
        # mol = self.get_mol(index)  # New
        # mol = Chem.MolFromSmiles(mol)
        # mol = AllChem.AddHs(mol)
        pos = self.get_pos(index)  # New
        # print(f'pos',pos)
        # pos_list_of_lists = ast.literal_eval(pos)
        # pos = np.array(pos_list_of_lists)
        return image, image_false, smiles, smiles_false, k_100, k_1000, k_10000, pos  # New
        # print(f'pos', type(pos))
        # print(f"mol",mol)
        # data = self.get_data(index)
        # return image, image_false, smiles, smiles_false, k_100, k_1000, k_10000 ,data # New

    def ocsr_predict_getItem(self, index):
        image = self.get_image(index)
        smiles = self.get_smiles(index)
        image_id = self.all_image_id[index]
        return smiles, image, int(image_id)

    def ocsr_getItem(self, index):
        image = self.get_image(index)
        smiles = self.get_oscr_smiles(index)
        image_id = self.all_image_id[index]
        image_false = self.get_false_image()
        return image, image_false, smiles, int(image_id)

    def get_num_to_modify(self, length, pn0=0.1, pn1=0.3, pn2=0.5, pn3=0.8):
        prob = random.random()
        if prob < pn0:
            num_to_modify = 1
        elif prob < pn1:
            num_to_modify = 2
        elif prob < pn2:
            num_to_modify = 3
        elif prob < pn3:
            num_to_modify = 4
        else:
            num_to_modify = 5

        if length <= 4:
            num_to_modify = min(num_to_modify, 1)
        else:
            num_to_modify = min(num_to_modify, length // 2)
        return num_to_modify

    def change_smiles(self, smiles, pt0=0.25, pt1=0.5, pt2=0.75):
        length = len(smiles)
        num_to_modify = self.get_num_to_modify(length)

        raw_chars = re.findall(self.regex, smiles)
        chars = []
        index = np.arange(0, length)
        random.shuffle(index)
        index = index[: num_to_modify]

        for i, t in enumerate(raw_chars):
            if i not in index:
                chars.append(t)
            else:
                prob = random.random()
                randomAotm = raw_chars[random.randint(0, len(raw_chars) - 1)]
                if prob < pt0:
                    chars.append(randomAotm)
                elif prob < pt1:
                    chars.append(randomAotm)
                    chars.append(t)
                elif prob < pt2:
                    chars.append(t)
                    chars.append(t)
                else:
                    continue

        new_smiles = ''.join(chars[: 202 - 1])
        return new_smiles

    def filter_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if smiles != None and mol is not None and mol.GetNumHeavyAtoms() > 0:
            return smiles


def combine_features(x, pos):
    """
    将节点特征和3D坐标数据结合起来。

    Args:
        x (torch.Tensor): 节点特征矩阵，维度为 [num_nodes, num_node_features]
        pos (torch.Tensor): 3D坐标矩阵，维度为 [num_nodes, 3]

    Returns:
        torch.Tensor: 融合后的特征矩阵，维度为 [num_nodes, num_node_features + 3]
    """
    return torch.cat([x, pos], dim=1)


def smi2_3Dcoords(smi, cnt=1):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)

    coordinate_list = []
    for seed in range(cnt):
        try:
            res = AllChem.EmbedMolecule(mol, randomSeed=seed)
            if res == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol)
                    coordinates = mol.GetConformer().GetPositions()
                except:
                    print("Failed to generate 3D, replace with 2D")
                    coordinates = smi2_2Dcoords(smi)
            elif res == -1:
                mol_tmp = Chem.MolFromSmiles(smi)
                AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000, randomSeed=seed)
                mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
                try:
                    AllChem.MMFFOptimizeMolecule(mol_tmp)
                    coordinates = mol_tmp.GetConformer().GetPositions()
                except:
                    print("Failed to generate 3D, replace with 2D")
                    coordinates = smi2_2Dcoords(smi)
        except:
            print("Failed to generate 3D, replace with 2D")
            coordinates = smi2_2Dcoords(smi)

        assert len(mol.GetAtoms()) == len(coordinates), "3D coordinates shape is not align with {}".format(smi)
        coordinate_list.append(coordinates.astype(np.float32))
    return coordinate_list


def smi2_2Dcoords(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    assert len(mol.GetAtoms()) == len(coordinates), "2D coordinates shape is not align with {}".format(smi)
    return coordinates


def inner_smi2coords(content):
    smi = content
    cnt = 1  # 只生成一个3D坐标
    mol = Chem.MolFromSmiles(smi)
    if len(mol.GetAtoms()) > 400:
        coordinate_list = [smi2_2Dcoords(smi)] * (cnt + 1)
        print("atom num >400, use 2D coords", smi)
    else:
        coordinate_list = smi2_3Dcoords(smi, cnt)
        coordinate_list.append(smi2_2Dcoords(smi).astype(np.float32))
    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return {'atoms': atoms, 'coordinates': coordinate_list, 'smi': smi}


def smi2coords(content):
    try:
        return inner_smi2coords(content)
    except:
        print("Failed smiles: {}".format(content))
        return None


def process_data(smiles, pos, smiles2graph, isomorphic_core):
    # smiles = row['smiles']
    mol = Chem.MolFromSmiles(smiles)
    mol = AllChem.AddHs(mol)

    # pos = smi2_3Dcoords(smiles)[0]  # 获取3D坐标
    data = Data()

    # mol = Chem.MolFromSmiles(smiles)
    # mol = AllChem.AddHs(mol)

    # pos = smi2_3Dcoords(smiles)[0]  # 获取3D坐标

    # data.smiles = smiles

    graph = smiles2graph(mol)

    assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
    assert len(graph["node_feat"]) == graph["num_nodes"]

    data.__num_nodes__ = int(graph["num_nodes"])
    data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
    data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
    data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
    data.y = None

    data.ring_mask = torch.from_numpy(graph["ring_mask"]).to(torch.bool)
    data.ring_index = torch.from_numpy(graph["ring_index"]).to(torch.int64)
    data.nf_node = torch.from_numpy(graph["nf_node"]).to(torch.int64)
    data.nf_ring = torch.from_numpy(graph["nf_ring"]).to(torch.int64)
    data.num_rings = int(graph["num_rings"])
    data.n_edges = int(graph["n_edges"])
    data.n_nodes = int(graph["n_nodes"])
    data.n_nfs = int(graph["n_nfs"])

    data.pos = torch.from_numpy(pos).to(torch.float)
    data.rdmol = deepcopy(mol)

    data.nei_src_index = torch.from_numpy(graph["nei_src_index"]).to(torch.int64)
    data.nei_tgt_index = torch.from_numpy(graph["nei_tgt_index"]).to(torch.int64)
    data.nei_tgt_mask = torch.from_numpy(graph["nei_tgt_mask"]).to(torch.bool)

    data.isomorphisms = isomorphic_core(mol)

    data.x = combine_features(data.x, data.pos)
    # Add batch and num_graphs attributes
    data.batch = torch.zeros(data.num_nodes, dtype=torch.int64)
    data.num_graphs = 1

    return data


'''

#原版没有mol和pos
class BaseDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 image_size: int,
                 names: list,
                 transforms,
                 smiles_column_name: str = "Smiles",
                 fingerprint_column_name: str = "Fingerprint",
                 k_100_column_name: str = 'k_100',
                 k_1000_column_name: str =  'k_1000',#'idd2k_1000',
                 k_10000_column_name: str = 'k_10000',# 'idd2k_10000',
                 label_name = None,
                 split = 'train',
                 is_pretrain=True,
                 MoleculeNet = True,
                 ocsr = False
            ):
        self.image_size = image_size
        self.smiles_column_name = smiles_column_name
        self.names = names
        self.data_dir = data_dir
        self.split = split
        self.transforms = transforms
        self.is_pretrain = is_pretrain
        # self._x = Xeger()
        self.regex = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        if len(names) != 0:
            # tables = [
            #     pa.ipc.RecordBatchFileReader(
            #         pa.memory_map(f"{data_dir}/{name}.arrow", "r")
            #     ).read_all()
            #     for name in names
            #     if os.path.isfile(f"{data_dir}/{name}.arrow")
            # ]
            print(f"{data_dir}/{names[0]}.arrow")
            tables = [
                pa.ipc.RecordBatchFileReader(
                    pa.memory_map(f"{data_dir}/{name}.arrow", "r")
                ).read_all()
                for name in names
                if os.path.isfile(f"{data_dir}/{name}.arrow")
            ]

            self.table = pa.concat_tables(tables, promote=True)
            #smiles_column_name="smiles"

            # Pretraining
            if self.is_pretrain and smiles_column_name != "" and k_100_column_name!="":
                self.smiles_column_name = smiles_column_name
                self.fingerprint_column_name = fingerprint_column_name
                self.k_100_column_name = k_100_column_name
                self.all_smiles = self.table[smiles_column_name].to_pandas().tolist()
                self.all_k_100 = self.table[k_100_column_name].to_pandas().tolist()
                self.all_k_1000 = self.table[k_1000_column_name].to_pandas().tolist()
                self.all_k_10000 = self.table[k_10000_column_name].to_pandas().tolist()

            # MoleculeNet downstream
            elif not self.is_pretrain and smiles_column_name != "" and MoleculeNet == True:
                self.smiles_column_name = smiles_column_name
                self.all_smiles = self.table[smiles_column_name].to_pandas().tolist()
                self.labels = self.table[label_name].to_pandas().tolist()

            # ocsr downstream
            elif not self.is_pretrain and smiles_column_name != "" and ocsr == True:
                self.smiles_column_name = smiles_column_name
                self.all_smiles = self.table[smiles_column_name].to_pandas().tolist()
                self.all_image_id = self.table['image_id'].to_pandas().tolist()
            # error
            else:
                self.all_smiles = list()
                self.all_fingerprint = list()
                self.all_k_100 = list()
                self.all_k_1000 = list()
                self.all_k_10000 = list()
        # error
        else:
            self.all_smiles = list()
            self.all_fingerprint = list()
            self.all_k_100 = list()
            self.all_k_1000 = list()
            self.all_k_10000 = list()

    def get_raw_image(self, index, image_key="images"):
#获取原始图像，利用索引读取原始图像
        image_bytes = io.BytesIO(self.table[image_key][index].as_py())
        image_bytes.seek(0)
        return Image.open(image_bytes).convert("RGB")

    def get_image(self, index, image_key="images"):
#用于获取处理后的图像数据的方法
        image = self.get_raw_image(index, image_key=image_key)
        image_tensor = self.transforms(image)
        return image_tensor

    def get_false_image(self, image_key="images"):
#获取随机选择的图像数据的方法，可用于负样本的生成。
        random_index = random.randint(0, len(self.all_smiles) - 1)
        image = self.get_raw_image(random_index, image_key=image_key)
        image_tensor = self.transforms(image)
        return image_tensor

    def get_smiles(self, index):#fintune
        smiles = self.all_smiles[index][0]
        #print(f'smiles',smiles)
        # mol = Chem.MolFromSmiles(smiles)
        # smiles = Chem.MolToSmiles(mol, doRandom=True)  # 不同表示
        return smiles
    def get_smile(self, index):#pretrain
        smiles = self.all_smiles[index]
        #print(f'smiles',smiles)
        # mol = Chem.MolFromSmiles(smiles)
        # smiles = Chem.MolToSmiles(mol, doRandom=True)  # 不同表示
        return smiles
    def get_false_smiles(self,index):
        raw_smiles = self.all_smiles[index]
        smiles_false = self.change_smiles(raw_smiles)
        return smiles_false

    def get_k_100(self,index):
        k_100 = self.all_k_100[index]# self.all_k_100[index][0]
        return k_100

    def get_k_1000(self,index):
        k_1000 = self.all_k_1000[index]# self.all_k_1000[index][0]
        return k_1000

    def get_k_10000(self,index):
        k_1000 = self.all_k_10000[index]# self.all_k_10000[index][0]
        return k_1000

    def get_label(self,index):
        assert not self.is_pretrain,"now,pretraining!!just download stream tasks have labels."
        return int(self.labels[index]) # self.labels[index][0] | self.labels[index]

    def get_labels(self):
        labels = np.array(self.labels)
        #return labels[:,0].tolist()  # 下游微调
        #return labels.tolist()
        #return labels
        return self.labels   # EGFR

    #根据给定的索引从数据集中获取指定样本的 SMILES 表示，但对于 OCSR 表示数据，会首先将原始 SMILES 表示转换为 RDKit Mol 对象，然后再重新生成 SMILES 表示。
    def get_oscr_smiles(self, index):
        smiles = self.all_smiles[index]
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol, doRandom=True)
        return smiles

    def __len__(self):
        return len(self.all_smiles)

    def getItem(self, index):
        image = self.get_image(index)
        image_false = self.get_false_image()
        # smiles = '' # self.get_smiles(index)
        #smiles = self.get_smiles(index)
        smiles = self.get_smile(index)
        k_100 = self.get_k_100(index)
        k_1000 = self.get_k_1000(index)
        k_10000 = self.get_k_10000(index)
        smiles_false = self.get_false_smiles(index)
        #label = self.get_label(index)
        # print(smiles,k_100,k_1000,k_10000)
        return image,image_false,smiles,smiles_false,k_100,k_1000,k_10000

    def ocsr_predict_getItem(self,index):
        image = self.get_image(index)
        smiles = self.get_smiles(index)
        image_id = self.all_image_id[index]
        return smiles,image,int(image_id)

    def ocsr_getItem(self,index):
        image = self.get_image(index)
        smiles = self.get_oscr_smiles(index)
        image_id = self.all_image_id[index]
        image_false = self.get_false_image()
        return image,image_false,smiles,int(image_id)

    def get_num_to_modify(self, length, pn0=0.1, pn1=0.3, pn2=0.5,pn3=0.8):
#根据长度和一组概率参数返回一个修改数量。
        prob = random.random()
        if prob < pn0: num_to_modify = 1
        elif prob < pn1: num_to_modify = 2
        elif prob < pn2: num_to_modify = 3
        elif prob < pn3: num_to_modify = 4
        else: num_to_modify = 5

        if length <= 4: num_to_modify = min(num_to_modify, 1)
        else: num_to_modify = min(num_to_modify, length // 2)  # smaller than length // 2
        return num_to_modify

    def change_smiles(self, smiles,pt0=0.25, pt1=0.5, pt2=0.75):

        length = len(smiles)
        num_to_modify = self.get_num_to_modify(length)

        raw_chars = re.findall(self.regex,smiles)
        chars = []
        index = np.arange(0, length)
        random.shuffle(index)
        index = index[: num_to_modify]

        for i, t in enumerate(raw_chars):
            if i not in index: chars.append(t)
            else:
                prob = random.random()
                randomAotm = raw_chars[random.randint(0,len(raw_chars)-1)]
                # randomAotm = self._x.xeger(self.regex)
                if prob < pt0: # replace
                    chars.append(randomAotm)
                elif prob < pt1: # insert
                    chars.append(randomAotm)
                    chars.append(t)
                elif prob < pt2: # insert
                    chars.append(t)
                    chars.append(t)
                else: # delete
                    continue

        new_smiles = ''.join(chars[: 202-1])
        return new_smiles

    def filter_smiles(self, smiles):

        mol = Chem.MolFromSmiles(smiles)
        if smiles != None and mol is not None and mol.GetNumHeavyAtoms() > 0:
            #print(smiles)
            return smiles




'''



