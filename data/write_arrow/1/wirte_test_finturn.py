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
# smiles_image_pairs/5w_data.json

def read_image(id,image_folder_path,split):
    path = image_folder_path + str('/') + split  + str('/') + str(id) + str('.png')
    print(path)
    with open(path, "rb") as fp:
        binary = fp.read()
    return binary

def make_arrow(csv_path, image_folder_path,dataset_root,split = 'train'):
    df = pd.read_csv(csv_path)
    print(df.columns)
    df.drop(['Unnamed: 0', 'unnamed: 0.2', 'unnamed: 0.1', 'unnamed: 0','unnamed: 0.2.1', 'unnamed: 0.1.1', 'unnamed: 0.3',
    'scaffolds', 'scaffoldcount', 'molweight', 'heavyatomcount', 'sascore','logp', 'penalizedlogp'],axis=1,inplace=True)

    df.columns = ['Smiles','image_id']
    df['images'] = df['image_id'].apply(read_image,args=(image_folder_path,split))

    write_table(df, dataset_root, split)

def write_table(df,dataset_root,split):
    table = pa.Table.from_pandas(df)
    with pa.OSFile(
            f"{dataset_root}/{split}.arrow", "wb"
    ) as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)
    print('donw!!')
# #
#
# make_arrow(csv_path='/root/zx/data/600w/val_386883.csv',
#            image_folder_path = '/root/zx/data/600w/images',
#             dataset_root='/root/zx/data/600w/',
#            split = 'val'
#            )
#
# make_arrow(csv_path='/root/zx/data/600w/test_383820.csv',
#            image_folder_path = '/root/zx/data/600w/images',
#             dataset_root='/root/zx/data/600w/',
#            split = 'test'
#            )

# make_arrow(csv_path='/data1/zx/data/ocsr_finturn/train_31w.csv',
#            image_folder_path = '/data1/zx/data/ocsr_finturn/image',
#            dataset_root='/data1/zx/data/ocsr_finturn',
#            split = 'train'
#            )

# make_arrow(csv_path='/data1/zx/data/ocsr_finturn/test_7w.csv',
#            image_folder_path = '/data1/zx/data/ocsr_finturn/image',
#            dataset_root='/data1/zx/data/ocsr_finturn',
#            split = 'test'
#            )


#######################
# 处理10W域外骨架测试集
#
######################
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")

def filter_smiles(smiles):
    input_ids = tokenizer.encode(smiles)
    max_ids = max(input_ids)
    if max_ids < 128:
        return smiles
    else:
        return float('nan')

def read_image_Oom(id,image_folder_path):
    path = image_folder_path + str('/') + str(id) + str('.png')
    print(path)
    with open(path, "rb") as fp:
        binary = fp.read()
    return binary

def make_arrow_Oom(csv_path, image_folder_path,dataset_root,split = 'train'):
    print(image_folder_path)
    df = pd.read_csv(csv_path)
    df.drop(['Unnamed: 0', 'unnamed: 0', 'label', 'fps', 'filename', 'scaffolds', 'scaffoldcount',
       'molweight', 'heavyatomcount', 'sascore', 'logp', 'penalizedlogp',],axis=1,inplace=True)
    # "images", "Smiles", "image_id", "Fingerprint", "k_100", "k_1000", "k_10000"

    df.columns = ['k_100', 'k_1000', 'k_10000', 'smiles', 'image_id']
    df["Smiles"] = df["smiles"].map(filter_smiles)
    df = df.drop(["smiles"], axis=1)
    df = df.dropna(subset=["Smiles"])

    df['images'] = df['image_id'].apply(read_image_Oom,args=(image_folder_path,))

    write_table(df, dataset_root, split)

# make_arrow_Oom(csv_path='/data1/zx/data/968W/ScaffoldCount10_10w.csv',
#            image_folder_path = '/data1/zx/data/ScaffoldCount10/images',
#            dataset_root='/data1/zx/data/ScaffoldCount10',
#            split = 'test'
#            )

make_arrow_Oom(csv_path='/data1/zx/data/968W/ScaffoldCount1_10w.csv',
           image_folder_path = '/data1/zx/data/ScaffoldCount1/images',
           dataset_root='/data1/zx/data/ScaffoldCount1',
           split = 'test'
           )