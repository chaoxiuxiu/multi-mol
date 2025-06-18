import pandas
import os
import re
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import json
import torch
'''
    smiles2images from csv
    path:csv path
    out_dir:output dir
'''
def smiles2image_save_as_json(path,image_out_dir,json_out_dir,json_name,start_id=0):
    out_path = os.path.join(os.getcwd(),image_out_dir)
    os.makedirs(out_path, exist_ok=True)

    if path.split('.')[-1] not in ['csv','txt']:
        raise NotImplementedError

    df = pd.read_csv(path)
    smi_list = df['Smiles']
    df['image_id'] = range(start_id,start_id + len(smi_list.values))

    for index, smi in enumerate(smi_list):
        mol = Chem.MolFromSmiles(smi)
        Draw.MolToFile(mol, f'{out_path}/{df["image_id"][index]}.png')
        if index % 10000 == 0:
            print(f'{index // 10000}w is done!')
    # update csv
    print("smiles2image is done!!")
    json_obj = eval(df.to_json(orient='records'))
    with open(f'{json_out_dir}/{json_name}.json', 'w', encoding='utf-8') as file:
        file.write(json.dumps(json_obj, indent=2, ensure_ascii=False))

'''
    分子筛选
    MW : 在12-600之间
    Heavy Atom Count : 3-50
    logP: -7 到 5
    
    :params
    df:DataFrame Object,
    output_dir:path
    output_file_name:str
'''
def evaluate(df,output_dir,output_file_name):
    df_ok = df[
        df.MolWeight.between(*[12, 600]) &  # MW
        df.logP.between(*[-7, 5]) &  # LogP
        df.heavyAtomCount.between(*[3, 50])  # HeavyAtomCount
    ]
    return df_ok

'''
csv_file->json_file
json_obj = csv2json(raw_data,'./','5w_data')
'''
def csv2json(df,out_dir,json_name):
    json_obj = eval(df.to_json(orient='records'))
    with open(f'{out_dir}/{json_name}.json', 'w', encoding='utf-8') as file:
        file.write(json.dumps(json_obj, indent=2, ensure_ascii=False))
    return json_obj

# 规范化tokenizer
def smi_tokenizer(smi):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    try:
        assert re.sub('\s+', '', smi) == ''.join(tokens)
    except:
        return ''
    return ' '.join(tokens)

#
def getDict(path,out_dir):
    out_path = os.path.join(os.getcwd(),out_dir)
    os.makedirs(out_path, exist_ok=True)
    if path.split('.')[-1] not in ['csv','txt','tsv']:
        raise NotImplementedError

    dic = dict()
    df = pd.read_csv(path)
    smi_list = df['Smiles'].values.tolist()
    smi_list = [smi_tokenizer(x) for x in smi_list]
    for i,smi in enumerate(smi_list):
        atoms = smi.split(' ')
        for i in atoms:
            if i in dic.keys():
                dic[i] += 1
            else:
                dic[i] = 1
    dic = sorted(dic.items(),key = lambda x:x[1],reverse = True)
    print(dic)
    return dic


def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


def smiles2image_save_as_json_by_one_file(path,image_out_dir,json_out_dir,json_name,start_id=0):
    out_path = os.path.join(os.getcwd(),image_out_dir)
    os.makedirs(out_path, exist_ok=True)

    if path.split('.')[-1] not in ['csv','txt','tsv']:
        raise NotImplementedError
    if path.split('.')[-1] in ['tsv']:  # moleculeNet
        df = pd.read_csv(path,sep = '\t')
        smi_list = df['text_a']
    else:
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower()
        smi_list = df['smiles']
    df['image_id'] = df.index

    s = 0
    for index, smi in enumerate(smi_list):
        mol = Chem.MolFromSmiles(smi)
        if mol==None:
            continue
        Draw.MolToFile(mol, f'{out_path}/{df["image_id"][index]}.png')
        s += 1
        if index % 1000 == 0:
            print(f'{index // 1000}k is done!')

    # update csv
    print(f"smiles2image is done!! Total: {s}")
    json_obj = eval(df.to_json(orient='records'))
    with open(f'{json_out_dir}/{json_name}.json', 'w', encoding='utf-8') as file:
        file.write(json.dumps(json_obj, indent=2, ensure_ascii=False))
