"""Functions that can be used to preprocess SMILES sequnces in the form used in the publication."""
import numpy as np
import pandas as pd
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit import Chem
from rdkit.Chem import Descriptors
REMOVER = SaltRemover()
ORGANIC_ATOM_SET = set([5, 6, 7, 8, 9, 15, 16, 17, 35, 53])

def randomize_smile(sml):#随机排列一个 SMILES 序列中原子的顺序
    """Function that randomizes a SMILES sequnce. This was adapted from the
    implemetation of E. Bjerrum 2017, SMILES Enumeration as Data Augmentation
    for Neural Network Modeling of Molecules.
    Args:
        sml: SMILES sequnce to randomize.
    Return:
        randomized SMILES sequnce or
        nan if SMILES is not interpretable.
    接受一个 SMILES 序列作为输入，并尝试将其转换为 RDKit 分子对象。然后，它随机重排原子的顺序，并返回重排后的 SMILES 序列。如果输入的 SMILES 序列无法解析成分子对象，则返回 NaN。
    """
    try:
        m = Chem.MolFromSmiles(sml)
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m, ans)
        return Chem.MolToSmiles(nm, canonical=False)
    except:
        return float('nan')

def canonical_smile(sml):#返回一个 SMILES 序列的 RDKit 规范化的 SMILES 序列。
    """Helper Function that returns the RDKit canonical SMILES for a input SMILES sequnce.
    Args:
        sml: SMILES sequence.
    Returns:
        canonical SMILES sequnce."""
    return Chem.MolToSmiles(sml, canonical=True)

def keep_largest_fragment(sml):#返回一个 SMILES 序列中最大片段的 SMILES 序列。
    """Function that returns the SMILES sequence of the largest fragment for a input
    SMILES sequnce.

    Args:
        sml: SMILES sequence.
    Returns:
        canonical SMILES sequnce of the largest fragment.
    接受一个 SMILES 序列作为输入，然后通过 RDKit 获取其所有片段，并找到其中包含原子最多的片段作为最大片段，并返回其 SMILES 序列。
    """
    mol_frags = Chem.GetMolFrags(Chem.MolFromSmiles(sml), asMols=True)
    largest_mol = None
    largest_mol_size = 0
    for mol in mol_frags:
        size = mol.GetNumAtoms()
        if size > largest_mol_size:
            largest_mol = mol
            largest_mol_size = size
    return Chem.MolToSmiles(largest_mol)

def remove_salt_stereo(sml, remover):
    """Function that strips salts and removes stereochemistry information from a SMILES.
    Args:
        sml: SMILES sequence.
        remover: RDKit's SaltRemover object.
    Returns:
        canonical SMILES sequnce without salts and stereochemistry information.
    接受一个 SMILES 序列和一个 RDKit 的 SaltRemover 对象作为输入，首先尝试使用 SaltRemover 对象去除盐，然后去除立体化学信息，并返回规范化的 SMILES 序列。如果无法解析输入的 SMILES 序列，则返回 NaN。
    """
    try:
        sml = Chem.MolToSmiles(remover.StripMol(Chem.MolFromSmiles(sml),
                                                dontRemoveEverything=True),
                               isomericSmiles=False)
        if "." in sml:
            sml = keep_largest_fragment(sml)
    except:
        sml = np.float("nan")
    return(sml)

def organic_filter(sml):
    """Function that filters for organic molecules.
    Args:
        sml: SMILES sequence.
    Returns:
        True if sml can be interpreted by RDKit and is organic.
        False if sml cannot interpreted by RDKIT or is inorganic.
    过滤有机分子。它接受一个 SMILES 序列作为输入，并尝试将其转换为 RDKit 分子对象。然后，检查分子中的原子是否全部属于有机原子集合，如果是则返回 True，否则返回 False。
    """
    try:
        m = Chem.MolFromSmiles(sml)
        atom_num_list = [atom.GetAtomicNum() for atom in m.GetAtoms()]
        is_organic = (set(atom_num_list) <= ORGANIC_ATOM_SET)
        if is_organic:
            return True
        else:
            return False
    except:
        return False

def filter_smiles(sml):
    '''
    用于过滤 SMILES 序列。它接受一个 SMILES 序列作为输入，并尝试将其转换为 RDKit 分子对象。然后，计算分子的分配系数、分子量、重原子数等属性，并对这些属性进行一系列限制条件的筛选。如果 SMILES 序列满足所有条件，则返回规范化的 SMILES 序列，否则返回 NaN。
    '''
    try:
        m = Chem.MolFromSmiles(sml)
        logp = Descriptors.MolLogP(m)
        mol_weight = Descriptors.MolWt(m)
        num_heavy_atoms = Descriptors.HeavyAtomCount(m)
        atom_num_list = [atom.GetAtomicNum() for atom in m.GetAtoms()]
        is_organic = set(atom_num_list) <= ORGANIC_ATOM_SET
        if ((logp > -5) & (logp < 7) &
            (mol_weight > 12) & (mol_weight < 600) &
            (num_heavy_atoms > 3) & (num_heavy_atoms < 50) &
            is_organic ):
            return Chem.MolToSmiles(m)
        else:
            return float('nan')
    except:
        return float('nan')
    
def get_descriptors(sml):
    '''
    用于从 SMILES 序列中提取分子描述符。它接受一个 SMILES 序列作为输入，并尝试将其转换为 RDKit 分子对象。然后，提取分子的一系列描述符，如 MolLogP、MolMR、BalabanJ、NumHAcceptors、NumHDonors、NumValenceElectrons 和 TPSA，并将其存储在列表中返回。如果无法解析输入的 SMILES 序列，则返回 NaN 列表。
    '''
    try:
        m = Chem.MolFromSmiles(sml)
        descriptor_list = []
        descriptor_list.append(Descriptors.MolLogP(m))
        descriptor_list.append(Descriptors.MolMR(m)) #ok
        descriptor_list.append(Descriptors.BalabanJ(m))
        descriptor_list.append(Descriptors.NumHAcceptors(m)) #ok
        descriptor_list.append(Descriptors.NumHDonors(m)) #ok
        descriptor_list.append(Descriptors.NumValenceElectrons(m))
        descriptor_list.append(Descriptors.TPSA(m)) # nice
        return descriptor_list
    except:
        return [np.float("nan")] * 7
def create_feature_df(smiles_df):
    '''
    用于创建包含分子描述符特征的 DataFrame。它接受一个包含 SMILES 序列的 DataFrame，并对其中的每个 SMILES 序列调用 get_descriptors 函数来提取分子描述符特征。然后，将特征存储在新的 DataFrame 中，并对特征进行标准化处理，最后将其与原始 DataFrame 进行连接并返回。
    '''
    temp = list(zip(*smiles_df['canonical_smiles'].map(get_descriptors)))
    columns = ["MolLogP", "MolMR", "BalabanJ", "NumHAcceptors", "NumHDonors", "NumValenceElectrons", "TPSA"]
    df = pd.DataFrame(columns=columns)
    for i, c in enumerate(columns):
        df.loc[:, c] = temp[i]
    df = (df - df.mean(axis=0, numeric_only=True)) / df.std(axis=0, numeric_only=True)
    df = smiles_df.join(df)
    return df

def preprocess_smiles(sml):
    """Function that preprocesses a SMILES string such that it is in the same format as
    the translation model was trained on. It removes salts and stereochemistry from the
    SMILES sequnce. If the sequnce correspond to an inorganic molecule or cannot be
    interpreted by RDKit nan is returned.

    Args:
        sml: SMILES sequence.
    Returns:
        preprocessd SMILES sequnces or nan.
    用于对 SMILES 序列进行预处理。它首先调用 remove_salt_stereo 函数去除盐和立体化学信息，然后调用 filter_smiles 函数进行分子过滤。最后返回预处理后的 SMILES 序列。
    """
    new_sml = remove_salt_stereo(sml, REMOVER)
    #new_sml = filter_smiles(new_sml)
    return new_sml


def preprocess_list(smiles,columns = 'text_a'):
    '''
    用于批量处理多个 SMILES 序列。它接受一个包含 SMILES 序列的列表或 DataFrame，并对其中的每个 SMILES 序列调用 preprocess_smiles 函数进行预处理。然后，将预处理后的 SMILES 序列存储在新的 DataFrame 中，并返回。
    '''
    df = pd.DataFrame(smiles)
    df["canonical_smiles"] = df[columns].map(preprocess_smiles)
    df = df.drop([columns], axis=1)
    df = df.dropna(subset=["canonical_smiles"])
    df = df.reset_index(drop=True)
    # df["random_smiles"] = df["canonical_smiles"].map(randomize_smile)
    # df = create_feature_df(df)
    return df
    