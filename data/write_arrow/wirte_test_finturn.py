#添加代码写入mol和pos两列
# 添加代码写入mol和pos两列
import json
import os
import pandas as pd
import pyarrow as pa
import random

from tqdm import tqdm
from glob import glob
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import MACCSkeys


def path2rest(path, iid2smiles):
    name = path.split("/")[-1].split('.png')[0]
    with open(path, "rb") as fp:
        binary = fp.read()
    smiles = iid2smiles[name]
    return [binary, smiles, name]


def read_image(id, image_folder_path, split):
    path = image_folder_path + str('/') + split + str('/') + str(id) + str('.png')
    print(path)
    with open(path, "rb") as fp:
        binary = fp.read()
    return binary


def make_arrow(csv_path, image_folder_path, dataset_root, split='train'):
    df = pd.read_csv(csv_path)
    print(df.columns)
    df.drop(
        ['Unnamed: 0', 'unnamed: 0.2', 'unnamed: 0.1', 'unnamed: 0', 'unnamed: 0.2.1', 'unnamed: 0.1.1', 'unnamed: 0.3',
         'scaffolds', 'scaffoldcount', 'molweight', 'heavyatomcount', 'sascore', 'logp', 'penalizedlogp'], axis=1,
        inplace=True)

    df.columns = ['Smiles', 'image_id', 'mol', 'pos']
    df['images'] = df['image_id'].apply(read_image, args=(image_folder_path, split))

    write_table(df, dataset_root, split)


def write_table(df, dataset_root, split):
    table = pa.Table.from_pandas(df)
    with pa.OSFile(f"{dataset_root}/{split}.arrow", "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)
    print('done!!')


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("E:/autodl-tmp/myCLIP/models/DeepChem/ChemBERTa-77M-MLM")


def filter_smiles(smiles):
    input_ids = tokenizer.encode(smiles)
    max_ids = max(input_ids)
    if max_ids < 128:
        return smiles
    else:
        return float('nan')


def read_image_Oom(id, image_folder_path):
    path = image_folder_path + str('/') + str(id) + str('.png')
    print(path)
    with open(path, "rb") as fp:
        binary = fp.read()
    return binary


def make_arrow_Oom(csv_path, image_folder_path, dataset_root, split='train'):
    print(image_folder_path)
    df = pd.read_csv(csv_path)
    # df.drop(['Unnamed: 0', 'unnamed: 0', 'label', 'fps', 'filename', 'scaffolds', 'scaffoldcount',
    #         'molweight', 'heavyatomcount', 'sascore', 'logp', 'penalizedlogp'], axis=1, inplace=True)
    df.drop(['Unnamed: 0', 'label', 'fps', 'filename', 'Scaffolds', 'ScaffoldCount',
             'MolWeight', 'heavyAtomCount', 'SAScore', 'logP', 'PenalizedLogP'], axis=1, inplace=True)

    # df.columns = ['k_100', 'k_1000', 'k_10000', 'smiles',  'pos',  'image_id']
    df.columns = ['k_100', 'k_1000', 'k_10000', 'smiles', 'image_id' , 'pos']
    df["Smiles"] = df["smiles"].map(filter_smiles)
    df = df.drop(["smiles"], axis=1)
    df = df.dropna(subset=["Smiles"])

    df['images'] = df['image_id'].apply(read_image_Oom, args=(image_folder_path,))

    write_table(df, dataset_root, split)

make_arrow_Oom(csv_path='E:/autodl-tmp/myCLIP/data/ADMETlab_data/Filtered_CycPeptMPDB_Peptide_Assay_PAMPA(4).csv',
               image_folder_path='E:/autodl-tmp/myCLIP/data/downstreamDataset/ADMETlab_data/CycPeptMPDB_Peptideall',
               dataset_root='E:/autodl-tmp/myCLIP/data/downstreamDataset/ADMETlab_data/CycPeptMPDB_Peptideall/',
               split='test')
