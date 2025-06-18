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

    for cap in tqdm(pairs):
        # filename = cap['image_id'].split("/")[-1]
        filename = str(cap['filename'])
        iid2smiles[filename].append(cap['Smiles'])
        iidfingerprint[filename].append(get_fingerprint(cap['Smiles']).ToBitString())
        idd2k_100[filename].append(cap['k_100'])

    paths = list(glob(f"/data3/djx/data/MolIS/img/*.png"))

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
        bs, columns=["images", "Smiles", "image_id", "Fingerprint", "k_100"],
    )

    table = pa.Table.from_pandas(dataframe)
    os.makedirs(dataset_root, exist_ok=True)
    with pa.OSFile(
        f"{dataset_root}/{split}.arrow", "wb"
    ) as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)

def to_json_file(path,json_out_dir,json_name,start_num,end_num=None):
    df = pd.read_csv(path)
    print(len(df))
    if end_num is not None:
        df = df[start_num:end_num]
    else:
        df = df[start_num:]
    print(len(df))
    df = df.dropna()
    print(len(df))
    json_obj = eval(df.to_json(orient='records'))
    with open(f'{json_out_dir}/{json_name}.json', 'w', encoding='utf-8') as file:
        file.write(json.dumps(json_obj, indent=2, ensure_ascii=False))
    print(f"{json_name} is down!")

# to_json_file('/data1/djx/data/MolIS/k_100/csv/100w_train_label.csv',json_out_dir='/data1/zx/data/100w',json_name='train_1',start_num=0,end_num=500000)
# to_json_file('/data1/djx/data/MolIS/k_100/csv/100w_train_label.csv','/data1/zx/data/100w','train_2',start_num=0)
# to_json_file('/data1/djx/data/MolIS/k_100/csv/10w_eval_label.csv','/data1/zx/data/100w','val')

#
# make_arrow(root='/data1/zx/data/100w',
#            data_split_path='train.json',
#            dataset_root = '/data1/zx/data/100W',
#            split = 'train'
#            )
#
# make_arrow(root='/data1/zx/data/10M_scaffold_split',
#            data_split_path='val.json',
#            dataset_root = '/data1/zx/data/100W',
#            split = 'val'
#            )
data = pd.read_csv('/data1/djx/data/djx/ImageGeneration/csv/968w_property_sorted_nonan.csv')
print(data.columns)
print(len(data))
# with open(f"/data1/zx/data/10M_scaffold_split/test_386560.json", "r") as fp:
#     pairs = json.load(fp)
#
# print(pairs[0])