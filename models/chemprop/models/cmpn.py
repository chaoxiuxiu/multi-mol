from argparse import Namespace
from typing import List, Union

import torch
import torch.nn as nn
import numpy as np

from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.nn_utils import index_select_ND, get_activation_function
import math
import torch.nn.functional as F
from torch_scatter import scatter_add
import pdb


class CMPNEncoder(nn.Module):
    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int):
        super(CMPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.atom_messages = args.atom_messages
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.args = args

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Input
        input_dim = self.atom_fdim
        self.W_i_atom = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        input_dim = self.bond_fdim
        self.W_i_bond = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        w_h_input_size_atom = self.hidden_size + self.bond_fdim
        self.W_h_atom = nn.Linear(w_h_input_size_atom, self.hidden_size, bias=self.bias)

        w_h_input_size_bond = self.hidden_size

        for depth in range(self.depth - 1):
            self._modules[f'W_h_{depth}'] = nn.Linear(w_h_input_size_bond, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear(
            (self.hidden_size) * 2,
            self.hidden_size)

        self.gru = BatchGRU(self.hidden_size)

        self.lr = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=self.bias)

        # add & concat functional group features
        if self.args.step != 'pretrain':
            self.cls = nn.Parameter(torch.randn(1, 133), requires_grad=True)
            self.W_i_atom_new = nn.Linear(self.atom_fdim * 2, self.hidden_size, bias=self.bias)

    def forward(self, step, mol_graph, features_batch=None) -> torch.FloatTensor:

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, atom_num, fg_num, f_fgs, fg_scope = mol_graph.get_components()  # 从 mol_graph 中提取图的组件，包括原子特征、键特征及其索引。
        if self.args.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb, f_fgs = (
                f_atoms.cuda(), f_bonds.cuda(),
                a2b.cuda(), b2a.cuda(), b2revb.cuda(), f_fgs.cuda())

        fg_index = [i * 13 for i in range(mol_graph.n_mols)]
        fg_indxs = [[i] * 133 for i in fg_index]
        fg_indxs = torch.LongTensor(fg_indxs).cuda()

        if self.args.step == 'functional_prompt':  # 使用功能提示生成器更新原子输入。
            # make sure the prompt exists
            assert self.W_i_atom.prompt_generator
            # Input
            input_atom = self.W_i_atom(f_atoms)  # num_atoms x hidden_size
            input_atom = self.W_i_atom.prompt_generator(input_atom, f_fgs, atom_num, fg_indxs)

        elif self.args.step == 'finetune_add':
            for i in range(len(fg_indxs)):
                f_fgs.scatter_(0, fg_indxs[i:i + 1], self.cls)  # 将分类参数 cls 散布到功能组特征 f_fgs。

            target_index = [val for val in range(mol_graph.n_mols) for i in range(13)]
            target_index = torch.LongTensor(target_index).cuda()
            fg_hiddens = scatter_add(f_fgs, target_index, 0)
            fg_hiddens_atom = torch.repeat_interleave(fg_hiddens, torch.tensor(atom_num).cuda(), dim=0)
            fg_out = torch.zeros(1, 133).cuda()
            fg_out = torch.cat((fg_out, fg_hiddens_atom), 0)
            f_atoms += fg_out
            # Input
            input_atom = self.W_i_atom(f_atoms)  # num_atoms x hidden_size

        elif self.args.step == 'finetune_concat':
            for i in range(len(fg_indxs)):
                f_fgs.scatter_(0, fg_indxs[i:i + 1], self.cls)  # 将分类参数 cls 散布到功能组特征 f_fgs。

            target_index = [val for val in range(mol_graph.n_mols) for i in range(13)]
            target_index = torch.LongTensor(target_index).cuda()
            fg_hiddens = scatter_add(f_fgs, target_index, 0)
            fg_hiddens_atom = torch.repeat_interleave(fg_hiddens, torch.tensor(atom_num).cuda(), dim=0)
            fg_out = torch.zeros(1, 133).cuda()
            fg_out = torch.cat((fg_out, fg_hiddens_atom), 0)
            f_atoms = torch.cat((fg_out, f_atoms), 1)
            # Input
            input_atom = self.W_i_atom_new(f_atoms)  # num_atoms x hidden_size使用新的线性层 W_i_atom_new 处理连接后的特征。

        else:
            # Input
            input_atom = self.W_i_atom(f_atoms)  # num_atoms x hidden_size直接使用线性层 W_i_atom 处理原子特征。

        input_atom = self.act_func(input_atom)
        message_atom = input_atom.clone()

        input_bond = self.W_i_bond(f_bonds)  # num_bonds x hidden_size
        message_bond = self.act_func(input_bond)
        input_bond = self.act_func(input_bond)
        # Message passing
        for depth in range(self.depth - 1):
            agg_message = index_select_ND(message_bond, a2b)
            agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
            message_atom = message_atom + agg_message

            # directed graph
            rev_message = message_bond[b2revb]  # num_bonds x hidden
            message_bond = message_atom[b2a] - rev_message  # num_bonds x hidden

            message_bond = self._modules[f'W_h_{depth}'](message_bond)
            message_bond = self.dropout_layer(self.act_func(input_bond + message_bond))

        agg_message = index_select_ND(message_bond, a2b)
        agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
        agg_message = self.lr(torch.cat([agg_message, message_atom, input_atom], 1))
        agg_message = self.gru(agg_message, a_scope)

        atom_hiddens = self.act_func(self.W_o(agg_message))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
            mol_vecs.append(cur_hiddens.mean(0))

        mol_vecs = torch.stack(mol_vecs, dim=0)
        return mol_vecs  # B x H返回每个分子的向量表示 mol_vecs。


class BatchGRU(nn.Module):
    def __init__(self, hidden_size=300):
        super(BatchGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True,
                          bidirectional=True)  # 创建一个双向 GRU 层，输入和输出的隐藏层大小为 hidden_size，并设置 batch_first=True 以支持批处理。
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.hidden_size),
                                1.0 / math.sqrt(self.hidden_size))  # 初始化一个偏置参数 bias，并用均匀分布初始化它。

    def forward(self, node, a_scope):
        hidden = node  # hidden 初始化为输入的 node 特征。
        message = F.relu(node + self.bias)  # 是对输入特征加上偏置后应用 ReLU 激活函数的结果。
        MAX_atom_len = max([a_size for a_start, a_size in a_scope])  # 计算批次中最大分子的原子数量 MAX_atom_len。
        # padding
        message_lst = []
        hidden_lst = []  # 初始化 message_lst 和 hidden_lst，用于存储消息和隐藏状态。
        for i, (a_start, a_size) in enumerate(a_scope):  # 对每个分子，提取其消息和隐藏状态。
            if a_size == 0:
                assert 0
            cur_message = message.narrow(0, a_start, a_size)
            cur_hidden = hidden.narrow(0, a_start, a_size)
            hidden_lst.append(cur_hidden.max(0)[0].unsqueeze(0).unsqueeze(0))

            cur_message = torch.nn.ZeroPad2d((0, 0, 0, MAX_atom_len - cur_message.shape[0]))(
                cur_message)  # 使用 torch.nn.ZeroPad2d 进行零填充，使所有消息长度相同。
            message_lst.append(cur_message.unsqueeze(0))

        message_lst = torch.cat(message_lst, 0)
        hidden_lst = torch.cat(hidden_lst, 1)
        hidden_lst = hidden_lst.repeat(2, 1, 1)  # 将隐藏状态复制一份以适应双向 GRU 的需求。
        cur_message, cur_hidden = self.gru(message_lst, hidden_lst)  # 将消息和隐藏状态传递给 GRU 层。

        # unpadding取消填充，将 GRU 输出的消息恢复到原始大小。
        cur_message_unpadding = []
        for i, (a_start, a_size) in enumerate(a_scope):
            cur_message_unpadding.append(cur_message[i, :a_size].view(-1, 2 * self.hidden_size))
        cur_message_unpadding = torch.cat(cur_message_unpadding, 0)

        message = torch.cat([torch.cat([message.narrow(0, 0, 1), message.narrow(0, 0, 1)], 1),
                             cur_message_unpadding], 0)  # 拼接并返回最终消息。
        return message

'''
class CMPN(nn.Module):
    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False):
        super(CMPN, self).__init__()
        self.args = args
        self.atom_fdim = atom_fdim or get_atom_fdim(args)  # 原子特征维度，如果未提供则通过 get_atom_fdim(args) 获取。
        self.bond_fdim = bond_fdim or get_bond_fdim(args) + \
                         (
                             not args.atom_messages) * self.atom_fdim  # * 2键特征维度，如果未提供则通过 get_bond_fdim(args) 获取。这个维度会根据是否使用原子消息进行调整。
        self.graph_input = graph_input  # 布尔值，指示输入是否为图表示。如果为 False，则需要将 SMILES 转换为图。
        self.encoder = CMPNEncoder(self.args, self.atom_fdim, self.bond_fdim)  # 作为编码器将分子图转换为向量表示。

    def forward(self, step, prompt: bool, smiles,
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        '''
        step：表示当前训练阶段（如预训练、微调等）。
        prompt：布尔值，指示是否使用提示。
        smiles：SMILES 表示的分子数据。
        features_batch：可选，附加的特征数组列表。
        '''
        if not self.graph_input:  # 如果只有特征，批次甚至都不会被使用
            batch = mol2graph(smiles, self.args, prompt)

        output = self.encoder.forward(step, batch, features_batch)
        return output


'''
class CMPN(nn.Module):
    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False):
        super(CMPN, self).__init__()
        self.args = args
        self.atom_fdim = atom_fdim or get_atom_fdim(args)#原子特征维度
        self.bond_fdim = bond_fdim or get_bond_fdim(args) + \
                            (not args.atom_messages) * self.atom_fdim # * 2
        self.graph_input = graph_input
        self.encoder = CMPNEncoder(self.args, self.atom_fdim, self.bond_fdim)

    def forward(self, step, prompt: bool, batch,
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        if not self.graph_input:  # if features only, batch won't even be used
            batch = mol2graph(batch, self.args, prompt)

        output = self.encoder.forward(step, batch, features_batch)
        return output
