#from data.write_arrow.utils import make_arrow
import json
import os
import pandas as pd
import pyarrow as pa
import random

from tqdm import tqdm
from glob import glob
from collections import defaultdict

def path2rest(path, iid2smiles, iid2labels):
    name = path.split("/")[-1].split('.png')[0]
    with open(path, "rb") as fp:
        binary = fp.read()
    smiles = iid2smiles[name]
    label = iid2labels[name]
    return [binary, smiles, label]


def make_arrow(root, data_split_path, dataset_root, split='train'):
    with open(f"{root}/{data_split_path}", "r") as fp:
        pairs = json.load(fp)
    print(f"Number of pairs in {data_split_path}: {len(pairs)}")

    iid2smiles = defaultdict(list)
    iid2labels = defaultdict(list)

    for cap in tqdm(pairs):
        filename = str(cap['image_id'])
        iid2smiles[filename].append(cap['smiles'])
        iid2labels[filename].append(cap['label'])

    paths = list(glob(f"{root}/images/{split}/*.png"))


    print(paths[0])
    #paths = list(glob(f"{root}/images/*.png"))
    random.shuffle(paths)

    print(f"Total images found in {split} set: {len(paths)}")

    smiles_paths = [path for path in paths if path.split("/")[-1][5:].split('.png')[0] in iid2smiles]

    if len(paths) == len(smiles_paths):
        print("All images have SMILES pair")
    else:
        print("Not all images have SMILES pair")
    print(f"Total images with SMILES pair: {len(smiles_paths)}")
    print(smiles_paths)
    #print(iid2smiles)
    bs = [path2rest(path, iid2smiles, iid2labels) for path in tqdm(smiles_paths)]
    bs = [entry for entry in bs if entry[1] is not None and entry[2] is not None]

    dataframe = pd.DataFrame(bs, columns=["images", "Smiles", 'labels'])

    table = pa.Table.from_pandas(dataframe)
    os.makedirs(dataset_root, exist_ok=True)
    with pa.OSFile(f"{dataset_root}/{split}.arrow", "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)
'''
make_arrow(root='/home/unionlab001/CXX/my/myCLIP/data/downstreamDataset/ADMETlab_data/CycPeptMPDB_Peptide',
           data_split_path='train.json',
           dataset_root = '/home/unionlab001/CXX/my/myCLIP/data/downstreamDataset/ADMETlab_data/CycPeptMPDB_Peptide',
           split = 'train'
           )

make_arrow(root='/home/unionlab001/CXX/my/myCLIP/data/downstreamDataset/ADMETlab_data/CycPeptMPDB_Peptide',
           data_split_path='test.json',
           dataset_root = '/home/unionlab001/CXX/my/myCLIP/data/downstreamDataset/ADMETlab_data/CycPeptMPDB_Peptide',
           split = 'test'
           )

make_arrow(root='/home/unionlab001/CXX/my/myCLIP/data/downstreamDataset/ADMETlab_data/CycPeptMPDB_Peptide',
           data_split_path='dev.json',
           dataset_root = '/home/unionlab001/CXX/my/myCLIP/data/downstreamDataset/ADMETlab_data/CycPeptMPDB_Peptide',
           split = 'dev'
           )
print('done!')

'''
make_arrow(root='E:/autodl-tmp/myCLIP/data/downstreamDataset/ADMETlab_data/CycPeptMPDB_Peptideall',
           data_split_path='test.json',
           dataset_root = 'E:/autodl-tmp/myCLIP/data/downstreamDataset/ADMETlab_data/CycPeptMPDB_Peptideall',
           split = 'test'
           )
# write_ADMETlab