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


def path2rest(path, iid2smiles, iidfingerprint,idd2k_100):
    name = path.split("/")[-1].split('.png')[0]
    with open(path, "rb") as fp:
        binary = fp.read()
    smiles = iid2smiles[name]
    fingerprint = iidfingerprint[name]
    k_100 = idd2k_100[name]
    return [binary, smiles, name, fingerprint, k_100]
# smiles_image_pairs/5w_data.json

def get_fingerprint(smiles_String):
    molecule = Chem.MolFromSmiles(smiles_String)
    fingerprint = MACCSkeys.GenMACCSKeys(molecule)
    return fingerprint




def make_arrow(root,data_split_path, dataset_root,split = 'train'):
    with open(f"{root}/{data_split_path}", "r") as fp:
        pairs = json.load(fp)

    iid2smiles = defaultdict(list)
    iidfingerprint = defaultdict(list)
    idd2k_100 = defaultdict(list)
    idd2k_1000 = defaultdict(list)
    idd2k_10000 = defaultdict(list)

    for cap in tqdm(pairs):
        # filename = cap['image_id'].split("/")[-1]
        filename = str(cap['filename'])
        iid2smiles[filename].append(cap['Smiles'])
        iidfingerprint[filename].append(get_fingerprint(cap['Smiles']).ToBitString())
        idd2k_100[filename].append(cap['k_100'])
        idd2k_1000[filename].append(cap['k_1000'])
        idd2k_10000[filename].append(cap['k_10000'])

    paths = list(glob(f"/data1/zx/data/10M_scaffold_split/images/{split}/*.png"))

    smiles_paths = [path for path in paths if path.split("/")[-1].split('.png')[0] in iid2smiles]

    if len(paths) == len(smiles_paths):
        print("all images have smiles pair")
    else:
        print("not all images have smiles pair")
    print(
        len(paths), len(smiles_paths),
    )

    bs = [path2rest(path, iid2smiles, iidfingerprint,idd2k_100) for path in tqdm(smiles_paths)]

    dataframe = pd.DataFrame(
        bs, columns=["images", "Smiles", "image_id", "Fingerprint", "k_100", "k_1000", "k_10000"],
    )


    table = pa.Table.from_pandas(dataframe)

    os.makedirs(dataset_root, exist_ok=True)
    with pa.OSFile(
        f"{dataset_root}/{split}.arrow", "wb"
    ) as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)