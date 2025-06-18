import collections
import re
import torch
import pandas as pd
from datasets.tokeinzer import smi_tokenizer
import json

class Vocab():
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<PAD>'] + ['<UNK>'] + ['<CLS>'] + ['<SEP>'] + ['<MASK>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):  #@save
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

# text file
if __name__=='__main__':
    path = '../data/5w_property.csv'
    df = pd.read_csv(path)
    smi_list = df['Smiles'].values.tolist()
    smi_list = [smi_tokenizer(x) for x in smi_list]
    atom_list = []
    for i, smi in enumerate(smi_list):
        atoms = smi.split(' ')
        atom_list.append(atoms)
    vocab = Vocab(atom_list)

    with open('vocab_index.txt', 'w') as f:
        for (key, value) in list(vocab.token_to_idx.items()):
            f.write(f"{key} {int(value)}\n")

    print(list(vocab.token_to_idx.items())[:20])

# Json file
# if __name__=='__main__':
#     path = '../data/5w_property.csv'
#     df = pd.read_csv(path)
#     smi_list = df['Smiles'].values.tolist()
#     smi_list = [smi_tokenizer(x) for x in smi_list]
#     atom_list = []
#     for i, smi in enumerate(smi_list):
#         atoms = smi.split(' ')
#         atom_list.append(atoms)
#     vocab = Vocab(atom_list)
#     dic = {}
#     with open('vocab.json', 'w') as f:
#         for (key, value) in list(vocab.token_to_idx.items()):
#             dic[key] = value
#
#         f.write(json.dumps(dic, ensure_ascii=False, indent=4, separators=(',', ':')))
#
#     print(list(vocab.token_to_idx.items())[:20])
