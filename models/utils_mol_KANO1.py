import math

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
import tqdm
import functools
import numpy as np
from torch.utils.data.distributed import DistributedSampler
# from models.model import ContrastiveLoss,CMCAN
# 初始化权重，正态分布，均值为0,偏差为0.02
# nn.Linear nn.embedding
from transformers import get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup, AdamW
from models.model import SCAN
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import torch
from torch_geometric.data import Batch
# from evaluate.metrics import RMSE


def cost_matrix_cosine(x, y, eps=1e-5):  # 计算了输入张量 x 和 y 之间的余弦距离矩阵。
    """Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]"""
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)  # 对输入张量 x 和 y 进行 L2 归一化，使它们的每个样本的特征向量都具有单位长度。
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))  # 使用 matmul 函数计算归一化后的 x 和 y 之间的内积矩阵，即余弦相似度矩阵。
    cosine_dist = 1 - cosine_sim  # 计算余弦距离矩阵，即将余弦相似度矩阵的值映射到 [0, 2] 区间，并通过减去 1 将其值映射到 [-1, 1] 区间。
    return cosine_dist


def trace(x):  # 计算输入张量的迹（trace），即输入张量的对角线元素之和。
    """ compute trace of input tensor (batched) """
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.bool, device=x.device).unsqueeze(0).expand_as(x)
    trace = x.masked_select(mask).contiguous().view(b, n).sum(dim=-1, keepdim=False)
    return trace


@torch.no_grad()
def ipot(C, x_len, x_pad, y_len, y_pad, joint_pad, beta, iteration, k):
    # 这个 ipot 函数实现了迭代最优输运算法（Iterative Optimal Transport）的计算过程，用于计算两个分布之间的最优输运。
    """ [B, M, N], [B], [B, M], [B], [B, N], [B, M, N]"""
    b, m, n = C.size()
    sigma = torch.ones(b, m, dtype=C.dtype, device=C.device) / x_len.unsqueeze(1)
    T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
    A = torch.exp(-C.transpose(1, 2) / beta)

    # mask padded positions
    sigma.masked_fill_(x_pad, 0)
    joint_pad = joint_pad.transpose(1, 2)
    T.masked_fill_(joint_pad, 0)
    A.masked_fill_(joint_pad, 0)

    # broadcastable lengths
    x_len = x_len.unsqueeze(1).unsqueeze(2)
    y_len = y_len.unsqueeze(1).unsqueeze(2)

    # mask to zero out padding in delta and sigma
    x_mask = (x_pad.to(C.dtype) * 1e4).unsqueeze(1)
    y_mask = (y_pad.to(C.dtype) * 1e4).unsqueeze(1)

    for _ in range(iteration):
        Q = A * T  # bs * n * m
        sigma = sigma.view(b, m, 1)
        for _ in range(k):
            delta = 1 / (y_len * Q.matmul(sigma).view(b, 1, n) + y_mask)
            sigma = 1 / (x_len * delta.matmul(Q) + x_mask)
        T = delta.view(b, n, 1) * Q * sigma
    T.masked_fill_(joint_pad, 0)
    return T


def optimal_transport_dist(
        txt_emb, img_emb, txt_pad, img_pad, beta=0.5, iteration=50, k=1
):  # 实现了计算文本嵌入和图像嵌入之间的最优输运距离的过程。
    """
    [B, M, D], [B, N, D], [B, M], [B, N]
    函数的输入参数包括：

        txt_emb：文本嵌入，形状为 [B, M, D]，其中 B 是批量大小，M 是文本嵌入的长度，D 是嵌入的维度。
        img_emb：图像嵌入，形状为 [B, N, D]，其中 B 是批量大小，N 是图像嵌入的长度，D 是嵌入的维度。
        txt_pad 和 img_pad：分别表示文本和图像的填充位置，形状分别为 [B, M] 和 [B, N]。
        beta：用于控制平滑度的参数。
        iteration：迭代次数。
        k：迭代中内部循环的次数。
    函数的主要步骤如下：

        使用 cost_matrix_cosine 函数计算文本嵌入和图像嵌入之间的余弦距离，并将填充位置的成本设为零。
        计算文本和图像的有效长度。
        调用 ipot 函数计算最优输运矩阵 T。
        使用 trace 函数计算最终的最优输运距离，并返回结果。
    """
    cost = cost_matrix_cosine(txt_emb, img_emb)
    # mask the padded inputs
    joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
    cost.masked_fill_(joint_pad, 0)

    txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)
    img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)

    T = ipot(
        cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad, beta, iteration, k
    )
    distance = trace(cost.matmul(T.detach()))
    return distance


def init_weights(module):
    '''
    这个 init_weights 函数用于初始化模型中的权重和偏置。它采用了一种比较常见的初始化方法，对于线性层和嵌入层，使用正态分布来初始化权重，均值为0，标准差为0.02。对于层归一化（LayerNorm）层，将偏置项初始化为零，权重初始化为1。如果线性层有偏置项，则将偏置项初始化为零。
    '''
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    # layerNorm,仿射变换1,0
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    #
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


# def set_schedule(pl_module):
#     lr = pl_module.hparams.config["learning_rate"]
#     wd = pl_module.hparams.config["weight_decay"]
#
#     no_decay = [
#         "bias",
#         "LayerNorm.bias",
#         "LayerNorm.weight",
#         "norm.bias",
#         "norm.weight",
#         "norm1.bias",
#         "norm1.weight",
#         "norm2.bias",
#         "norm2.weight",
#     ]
#     # head_names = [ "mlm_score", "itm_score", "mpp_score"]
#     head_names = ["mlm_score","itm_score","decoder_block"]
#
#     cross_modal_names = ['cross_modal']
#     lr_mult_head = pl_module.hparams.config["lr_mult_head"]
#     lr_mult_cross_modal = pl_module.hparams.config["lr_mult_cross_modal"]
#     end_lr = pl_module.hparams.config["end_lr"]
#     decay_power = pl_module.hparams.config["decay_power"]
#     optim_type = pl_module.hparams.config["optim_type"]
#     optimizer_grouped_parameters = [
#         {
#             "params": [
#                 p
#                 for n, p in pl_module.named_parameters()
#                 if not any(nd in n for nd in no_decay) # bias norm no-decay
#                 and not any(bb in n for bb in head_names) #
#                 and not any(ht in n for ht in cross_modal_names)
#             ],
#             "weight_decay": wd,
#             "lr": lr,
#             "name":[
#                 n
#                 for n, p in pl_module.named_parameters()
#                 if not any(nd in n for nd in no_decay) # bias norm no-decay
#                 and not any(bb in n for bb in head_names) #
#                 and not any(ht in n for ht in cross_modal_names)
#             ],
#         },
#         {
#             "params": [
#                 p
#                 for n, p in pl_module.named_parameters()
#                 if any(nd in n for nd in no_decay)
#                 and not any(bb in n for bb in head_names)
#                 and not any(ht in n for ht in cross_modal_names)
#             ],
#             "weight_decay": 0.0,
#             "lr": lr,
#             "name": [
#                 n
#                 for n, p in pl_module.named_parameters()
#                 if any(nd in n for nd in no_decay)
#                    and not any(bb in n for bb in head_names)
#                    and not any(ht in n for ht in cross_modal_names)
#             ],
#         },
#         {
#             "params": [
#                 p
#                 for n, p in pl_module.named_parameters()
#                 if not any(nd in n for nd in no_decay)
#                 and any(bb in n for bb in head_names)
#                 and not any(ht in n for ht in cross_modal_names)
#             ],
#             "weight_decay": wd,
#             "lr": lr * lr_mult_head,
#             "name": [
#                 n
#                 for n, p in pl_module.named_parameters()
#                 if not any(nd in n for nd in no_decay)
#                    and any(bb in n for bb in head_names)
#                    and not any(ht in n for ht in cross_modal_names)
#             ],
#         },
#         {
#             "params": [
#                 p
#                 for n, p in pl_module.named_parameters()
#                 if any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
#                 and not any(ht in n for ht in cross_modal_names)
#             ],
#             "weight_decay": 0.0,
#             "lr": lr * lr_mult_head,
#             "name": [
#                 n
#                 for n, p in pl_module.named_parameters()
#                 if any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
#                    and not any(ht in n for ht in cross_modal_names)
#             ],
#         },
#         {
#             "params": [
#                 p
#                 for n, p in pl_module.named_parameters()
#                 if not any(nd in n for nd in no_decay)
#                 and not any(bb in n for bb in head_names)
#                 and any(ht in n for ht in cross_modal_names)
#             ],
#             "weight_decay": wd,
#             "lr": lr * lr_mult_cross_modal,
#             "name": [
#                 n
#                 for n, p in pl_module.named_parameters()
#                 if not any(nd in n for nd in no_decay)
#                    and not any(bb in n for bb in head_names)
#                    and any(ht in n for ht in cross_modal_names)
#             ],
#         },
#         {
#             "params": [
#                 p
#                 for n, p in pl_module.named_parameters()
#                 if any(nd in n for nd in no_decay)
#                 and not any(bb in n for bb in head_names)
#                 and any(ht in n for ht in cross_modal_names)
#             ],
#             "weight_decay": 0.0,
#             "lr": lr * lr_mult_cross_modal,
#             "name": [
#                 n
#                 for n, p in pl_module.named_parameters()
#                 if any(nd in n for nd in no_decay)
#                    and not any(bb in n for bb in head_names)
#                    and any(ht in n for ht in cross_modal_names)
#             ],
#         },
#     ]
#     if optim_type == "adamw":
#         optimizer = AdamW(
#             optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
#         )
#     elif optim_type == "adam":
#         optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
#     elif optim_type == "sgd":
#         optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)
#
#     if pl_module.trainer.max_steps is None:
#         max_steps = (
#             len(pl_module.trainer.datamodule.train_dataloader())
#             * pl_module.trainer.max_epochs
#             // pl_module.trainer.accumulate_grad_batches
#         )
#     else:
#         max_steps = pl_module.trainer.max_steps
#
#     warmup_steps = pl_module.hparams.config["warmup_steps"]
#     if isinstance(pl_module.hparams.config["warmup_steps"], float):
#         warmup_steps = int(max_steps * warmup_steps)
#
#     if decay_power == "cosine":
#         scheduler = get_cosine_schedule_with_warmup(
#             optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps,
#         )
#     else:
#         scheduler = get_polynomial_decay_schedule_with_warmup(
#             optimizer,
#             num_warmup_steps=warmup_steps,
#             num_training_steps=max_steps,
#             lr_end=end_lr,
#             power=decay_power,
#         )
#
#     sched = {"scheduler": scheduler, "interval": "step"}
#
#     return (
#         [optimizer],
#         [sched],
#     )


def set_schedule(pl_module):
    '''
    这个 set_schedule 函数用于设置模型训练的优化器和学习率调度器。它根据配置文件中的参数设置不同部分的学习率，例如主体部分、头部部分 和跨模态部分。它还支持不同的优化器类型（如AdamW、Adam和SGD）和学习率调度器（如余弦退火调度器和多项式退火调度器）。

    主要步骤如下：

        根据配置文件中的参数获取学习率和权重衰减。
        根据模型的不同部分，将参数分组，为不同的参数组设置不同的学习率和权重衰减。
        根据优化器类型，创建相应的优化器对象（AdamW、Adam或SGD）。
        计算训练总步数和预热步数。
        根据学习率衰减策略创建学习率调度器对象（余弦退火或多项式退火）。
        最后，函数返回一个元组，包含了优化器列表和学习率调度器列表，用于 Trainer 对象的初始化。
    '''
    lr = pl_module.hparams.config["learning_rate"]
    wd = pl_module.hparams.config["weight_decay"]

    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ]
    # head_names = [ "mlm_score", "itm_score", "mpp_score"]
    head_names = ["mlm_score", "itm_score", "decoder_block", 'fpp_score','gtm_score'
                  # "fpp_score_smiles100","fpp_score_smiles100","fpp_score_smiles1000","fpp_score_smiles1000","fpp_score_smiles10000","fpp_score_smiles10000",
                  'MoleculeNet_classify_score']
    cross_modal_names = ['cross_modal']
    lr_mult_head = pl_module.hparams.config["lr_mult_head"]
    lr_mult_cross_modal = pl_module.hparams.config["lr_mult_cross_modal"]
    end_lr = pl_module.hparams.config["end_lr"]
    decay_power = pl_module.hparams.config["decay_power"]
    optim_type = pl_module.hparams.config["optim_type"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)  # bias norm no-decay
                   and not any(bb in n for bb in head_names)  #
                   and not any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": wd,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                   and not any(bb in n for bb in head_names)
                   and not any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": 0.0,
            "lr": lr,

        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                   and any(bb in n for bb in head_names)
                   and not any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult_head,

        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                   and any(bb in n for bb in head_names)
                   and not any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult_head,

        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                   and not any(bb in n for bb in head_names)
                   and any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult_cross_modal,

        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                   and not any(bb in n for bb in head_names)
                   and any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult_cross_modal,

        },
    ]
    if optim_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)
    elif optim_type == "rmsprop":
        optimizer = torch.optim.RMSprop(optimizer_grouped_parameters, lr=lr, alpha=0.99, eps=1e-8, momentum=0.9)
    elif optim_type == "adadelta":
        optimizer = torch.optim.Adadelta(optimizer_grouped_parameters, lr=lr, rho=0.9, eps=1e-6)

    if pl_module.trainer.max_steps is None:
        max_steps = (
                len(pl_module.trainer.datamodule.train_dataloader())
                * pl_module.trainer.max_epochs
                // pl_module.trainer.accumulate_grad_batches
        )
    else:
        max_steps = pl_module.trainer.max_steps

    warmup_steps = pl_module.hparams.config["warmup_steps"]
    if isinstance(pl_module.hparams.config["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )

    sched = {"scheduler": scheduler, "interval": "step"}

    return (
        [optimizer],
        [sched],
    )


def logger_mlm(pl_module, batch, data):  # 记录 MLM（Masked Language Modeling）任务的损失和准确率，并返回这些值以供进一步处理。
    images, _, smiles, *el = batch  # 从批量数据中获取图像、SMILES 序列和其他元素。
    infer_mlm = pl_module.infer(images, smiles, data= data, mask_smiles=True)  # 使用模型的 infer 方法生成 MLM 预测结果。
    mlm_logits = pl_module.mlm_score(infer_mlm["smiles_feats"])  # 通过 MLM 预测结果计算预测标签和损失。
    mlm_labels = infer_mlm['mlm_labels']
    loss_mlm = pl_module.focal_loss(
        mlm_logits.view(-1, pl_module.config["vocab_size"]),
        mlm_labels.view(-1),
    )
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mlm_loss")(loss_mlm)
    acc = getattr(pl_module, f"{phase}_mlm_accuracy")(
        mlm_logits, mlm_labels
    )
    pl_module.log(f"mlm/{phase}/loss", loss)  # 使用 pl_module.log 方法记录 MLM 损失和准确率，并打印准确率。
    pl_module.log(f"mlm/{phase}/accuracy", acc)
    print(f"mlm accuracy:{acc}")
    return {
        "loss_mlm": loss_mlm,
        "acc_mlm": acc,
    }


def logger_itm(pl_module, batch, data):
    images, image_false, smiles, smiles_false, *el, = batch  # 从批量数据中获取图像、假图像、SMILES 序列和假 SMILES 序列。

    # pos_len = images.size(0) // 2
    # neg_len = images.size(0) - pos_len
    # itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(images.device)
    # # 随机打乱
    # itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]
    # itm_images = torch.stack(
    #     [images[idx] if label == 1 else image_false[idx] for idx, label in enumerate(itm_labels)]).to(images.device)
    # infer_itm = pl_module.infer(itm_images, smiles)
    # itm_logits = pl_module.itm_score(infer_itm["cls_feats"])
    #
    # loss_itm = F.cross_entropy(itm_logits, itm_labels.long())

    infer_ocsr = pl_module.infer(images, smiles, image_false, smiles_false, data= data, change_smi=True)

    itm_logits = pl_module.itm_score(infer_ocsr["cls_feats"])  # 映射

    itm_labels = infer_ocsr['ocsr_labels'].to(images.device)
    loss_itm = pl_module.focal_loss(
        itm_logits,
        itm_labels.long()
    )

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_itm_loss")(loss_itm)  # Metrics.forwrad()
    acc = getattr(pl_module, f"{phase}_itm_accuracy")(
        itm_logits, itm_labels
    )
    pl_module.log(f"itm/{phase}/loss", loss)
    pl_module.log(f"itm/{phase}/accuracy", acc)
    print(f"itm accuracy:{acc}")
    return {
        "loss_itm": 8 * loss_itm,
        "acc_itm": acc,
    }

def logger_gtm(pl_module, batch, data):
    images, _, smiles, _, *el, = batch  # 从批量数据中获取图像、假图像、SMILES 序列和假 SMILES 序列。

    infer_ocsr = pl_module.infer(images, smiles,  data= data )

    gtm_logits = pl_module.gtm_score(infer_ocsr["gtm_embeddings"])  # 映射
    gtm_labels = infer_ocsr['gtm_labels'].to(images.device)
    loss_gtm = F.cross_entropy(gtm_logits, gtm_labels)
    #loss_itm = pl_module.focal_loss(itm_logits,itm_labels.long())

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_gtm_loss")(loss_gtm)  # Metrics.forwrad()
    acc = getattr(pl_module, f"{phase}_gtm_accuracy")(
        gtm_logits, gtm_labels
    )
    pl_module.log(f"gtm/{phase}/loss", loss)
    pl_module.log(f"gtm/{phase}/accuracy", acc)
    print(f"gtm loss:{loss}",f"gtm acc:{acc}")
    return {
        "loss_gtm": 4 * loss_gtm,
        "acc_gtm": acc,
    }


def logger_gcl(pl_module, batch, data):
    # images, image_false, smiles, smiles_false,*el, = batch  # 从批量数据中获取图像、假图像、SMILES 序列和假 SMILES 序列。
    images, _, smiles, *el = batch  # 从批量数据中获取图像、SMILES 序列和其他元素。

    infer_gcl = pl_module.infer(images, smiles, data= data)
    graph_emb1 = infer_gcl["graph_emb1"]
    graph_emb2 = infer_gcl["graph_emb2"]
    loss_gcl = pl_module.criterion(graph_emb1, graph_emb2)
    # loss = loss.sum() / mask.sum()mlm_logits
    # loss_sum += loss.item()
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_gcl_loss")(loss_gcl)
    pl_module.log(f"gcl/{phase}/loss", loss)
    print(f"gcl loss: {loss.item()}")
    return {
        "loss_gcl": 4 * loss_gcl,
    }


def contrastive_loss(embedding1, embedding2, margin=1.0):
    pos_dist = torch.sum((embedding1 - embedding2) ** 2, dim=1)
    neg_dist = torch.sum((embedding1.unsqueeze(1) - embedding2.unsqueeze(0)) ** 2, dim=2)
    neg_dist = neg_dist + (torch.eye(neg_dist.shape[0]) * 1e12).to(neg_dist.device)
    min_neg_dist = torch.min(neg_dist, dim=1)[0]
    loss = torch.mean(torch.relu(pos_dist - min_neg_dist + margin))
    return loss




def mask2d(pl_module, batch, data):
    images, _, smiles, *el = batch

    infer_ocsr = pl_module.infer(images, smiles, data= data)
    gt_x = data.x.to(images.device)
    gt_pos = data.pos.to(images.device)
    attr_mask_index = infer_ocsr["attr_mask_index"]  # 映射
    pred_attrs = infer_ocsr["pred_attrs"]  # 映射
    pos_predictions = infer_ocsr["pos_predictions"]
    pos_mask_idx = infer_ocsr["pos_mask_idx"]

    attr_mask_index = attr_mask_index.view(-1)
    pos_mask_idx = pos_mask_idx.view(-1)#展平成一维。

    attr_loss = 0
    pos_loss = 0
    for i in range(gt_x.shape[1]):
        pred = pred_attrs[i][attr_mask_index]#预测值
        gt = gt_x[:, i][attr_mask_index].to(images.device)#真实值
        attr_loss = attr_loss + F.cross_entropy(pred, gt, reduction="mean")#对于每个属性维度，提取预测值和真实值的掩码部分，计算交叉熵损失并累加到 attr_loss 中。
    #调用 update_iso_mask 更新图中掩盖位置的索引。新位置掩码索引 new_idx。
    new_idx = update_iso_mask(
        gt_pos,
        torch.where(pos_mask_idx.view(-1, 1), pos_predictions[-1], gt_pos),
        data,
        pos_mask_idx,
    )

    pos_mask_idx = pos_mask_idx.index_select(0, new_idx)#重新选择位置掩码索引
    gt = gt_pos[pos_mask_idx]#获取真实位置的掩码部分
    #获取真实位置的掩码部分，每个位置预测，计算真实位置与预测位置之间的 L2 范数损失，并按权重累加到 pos_loss 中
    for i, pos_pred in enumerate(pos_predictions):
        pred = pos_pred.index_select(0, new_idx)[pos_mask_idx]
        pos_loss = pos_loss + (gt - pred).norm(dim=-1).mean() * (
            1 if i == len(pos_predictions) - 1 else 0.1
        )

    loss_mask = attr_loss + pos_loss
    # loss_jem = contrastive_loss(img_embs, cap_embs,graph_embs)
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_m2d_loss")(loss_mask)  # Metrics.forwrad()

    pl_module.log(f"jem/{phase}/loss", loss)

    print(f"m2d loss: {loss.item()}")
    pl_module.log(f"m2d/{phase}/loss", loss)
    return {"loss_m2d": loss_mask}

#此函数在考虑图同构性的情况下更新节点位置的掩码。
def update_iso_mask(pos_y, pos_x, data, pos_mask):
    with torch.no_grad():
        pre_nodes = 0#pre_nodes 用于跟踪每个图中节点的起始索引。
        #对于每个图，从 data 中检索同构和节点数量。
        num_nodes = data.n_nodes
        isomorphisms = data.isomorphisms


        new_idx_x = []
        #获取当前图的同构信息和节点数，以及位置掩码。
        for i in range(data.num_graphs):
            #print(isomorphisms[i])

            current_isomorphisms = [
                torch.LongTensor(list(iso.values())).to(pos_x.device) for iso in isomorphisms[i]
            ]#iso.values() 会返回映射后的节点索引，这些是你需要用来进行后续操作的索引。
            cur_num_nodes = num_nodes[i]
            cur_pos_mask = pos_mask[pre_nodes : pre_nodes + cur_num_nodes]
            #如果没有同构或当前掩码全为 0，直接选择第一个同构索引
            if len(current_isomorphisms) == 1 or not torch.any(cur_pos_mask):
                new_idx_x.append(current_isomorphisms[0] + pre_nodes)
            #计算所有同构下的位置预测，选择最小损失的同构索引
            else:
                pos_y_i = pos_y[pre_nodes : pre_nodes + cur_num_nodes]
                pos_x_i = pos_x[pre_nodes : pre_nodes + cur_num_nodes]
                pos_x_list = []
                for iso in current_isomorphisms:
                    pos_x_list.append(torch.index_select(pos_x_i, 0, iso))

                total_iso = len(pos_x_list)
                pos_y_i = pos_y_i.repeat(total_iso, 1)
                pos_x_i = torch.cat(pos_x_list, dim=0)
                min_idx = mask_loss_one_graph(
                    pos_y_i, pos_x_i, cur_num_nodes, total_iso, cur_pos_mask
                )
                new_idx_x.append(current_isomorphisms[min_idx.item()] + pre_nodes)

            pre_nodes += cur_num_nodes#更新 pre_nodes 累计已处理的节点数

        return torch.cat(new_idx_x, dim=0)
#@staticmethod
#计算单个图中各个同构的损失
def mask_loss_one_graph(pos_y, pos_x, num_nodes, total_iso, pos_mask):
    #计算每个同构的损失，并返回损失最小的同构索引
    with torch.no_grad():
        loss = (pos_y - pos_x).norm(dim=-1, keepdim=True).view(-1, num_nodes).mean(-1)
        return torch.argmin(loss)


def conf2mol(pl_module, batch, data):
    images, _, smiles, *el = batch

    infer_ocsr = pl_module.infer(images, smiles,data= data,mode="conf2mol")
    gt_x = data.x.to(images.device)
    #attr_mask_index = None # 映射
    pred_attrs = infer_ocsr["pred_attrs"]  # 映射
    #pos_predictions = infer_ocsr["pos_predictions"]
    #pos_mask_idx = infer_ocsr["pos_mask_idx"]

    attr_loss = 0
    for i in range(gt_x.shape[1]):
        pred = pred_attrs[i]
        gt = gt_x[:, i]
        attr_loss = attr_loss + F.cross_entropy(pred, gt, reduction="mean")
    # loss_jem = contrastive_loss(img_embs, cap_embs,graph_embs)
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_c2m_loss")(attr_loss)  # Metrics.forwrad()

    pl_module.log(f"c2m/{phase}/loss", loss)

    print(f"c2m loss: {loss.item()}")
    pl_module.log(f"c2m/{phase}/loss", loss)
    return {"loss_c2m": attr_loss}

def mol2conf(pl_module, batch, data):
    images, _, smiles, *el = batch
    infer_ocsr = pl_module.infer(images, smiles, data= data, mode="mol2conf")
    gt_pos = data.pos.to(images.device)
    pos_predictions = infer_ocsr["pos_predictions"]
    pos_loss = 0
    data = data.to(images.device)
    new_idx = update_iso_mol2conf(gt_pos, pos_predictions[-1], data)
    for i, pos_pred in enumerate(pos_predictions):
        pos_loss = pos_loss + alignment_loss(
            gt_pos, torch.index_select(pos_pred, 0, new_idx), data
        ) * (1 if i == len(pos_predictions) - 1 else 0.1)
    # loss_jem = contrastive_loss(img_embs, cap_embs,graph_embs)
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_m2c_loss")(pos_loss)  # Metrics.forwrad()

    pl_module.log(f"m2c/{phase}/loss", loss)

    print(f"m2c loss: {loss.item()}")
    pl_module.log(f"m2c/{phase}/loss", loss)
    return {"loss_m2c": pos_loss}

def update_iso_mol2conf(pos_y, pos_x, batch):
    with torch.no_grad():
        pre_nodes = 0
        num_nodes = batch.n_nodes
        isomorphisms = batch.isomorphisms
        new_idx_x = []
        for i in range(batch.num_graphs):
            current_isomorphisms = [
                torch.LongTensor(list(iso.values())).to(pos_x.device) for iso in isomorphisms[i]
            ]
            cur_num_nodes = num_nodes[i]

            if len(current_isomorphisms) == 1:
                new_idx_x.append(current_isomorphisms[0] + pre_nodes)
            else:
                pos_y_i = pos_y[pre_nodes : pre_nodes + cur_num_nodes]
                pos_x_i = pos_x[pre_nodes : pre_nodes + cur_num_nodes]
                pos_y_mean = torch.mean(pos_y_i, dim=0, keepdim=True)
                pos_x_mean = torch.mean(pos_x_i, dim=0, keepdim=True)
                pos_x_list = []

                for iso in current_isomorphisms:
                    pos_x_list.append(torch.index_select(pos_x_i, 0, iso))

                total_iso = len(pos_x_list)
                pos_y_i = pos_y_i.repeat(total_iso, 1)
                pos_x_i = torch.cat(pos_x_list, dim=0)
                min_idx = mol2conf_loss_one_graph(
                    pos_y_i,
                    pos_x_i,
                    pos_y_mean,
                    pos_x_mean,
                    num_nodes=cur_num_nodes,
                    total_iso=total_iso,
                )
                new_idx_x.append(current_isomorphisms[min_idx.item()] + pre_nodes)
            pre_nodes += cur_num_nodes

        return torch.cat(new_idx_x, dim=0)

#@staticmethod
def mol2conf_loss_one_graph(pos_y, pos_x, pos_y_mean, pos_x_mean, num_nodes, total_iso):
    with torch.no_grad():
        total_nodes = pos_y.shape[0]
        y = pos_y - pos_y_mean
        x = pos_x - pos_x_mean
        a = y + x
        b = y - x
        a = a.view(-1, 1, 3)
        b = b.view(-1, 3, 1)
        tmp0 = torch.cat(
            [b.new_zeros((1, 1, 1)).expand(total_nodes, -1, -1), -b.permute(0, 2, 1)], dim=-1
        )
        eye = torch.eye(3).to(a).unsqueeze(0).expand(total_nodes, -1, -1)
        a = a.expand(-1, 3, -1)
        tmp1 = torch.cross(eye, a, dim=-1)
        tmp1 = torch.cat([b, tmp1], dim=-1)
        tmp = torch.cat([tmp0, tmp1], dim=1)
        tmpb = torch.bmm(tmp.permute(0, 2, 1), tmp).view(-1, num_nodes, 16)
        tmpb = torch.mean(tmpb, dim=1).view(-1, 4, 4)
        w, v = torch.linalg.eigh(tmpb)
        min_q = v[:, :, 0]
        rotation = quaternion_to_rotation_matrix(min_q)
        t = pos_y_mean - torch.einsum("kj,kij->ki", pos_x_mean.expand(total_iso, -1), rotation)
        #print(f"rotation", rotation.device, f"num_nodes", num_nodes.device)
        rotation = torch.repeat_interleave(rotation, num_nodes, dim=0)

        t = torch.repeat_interleave(t, num_nodes, dim=0)
        pos_x = torch.einsum("kj,kij->ki", pos_x, rotation) + t
        loss = (pos_y - pos_x).norm(dim=-1, keepdim=True).view(-1, num_nodes,).mean(-1)
        return torch.argmin(loss)

def quaternion_to_rotation_matrix(quaternion):
    q0 = quaternion[:, 0]
    q1 = quaternion[:, 1]
    q2 = quaternion[:, 2]
    q3 = quaternion[:, 3]

    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
    return torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=-1).reshape(-1, 3, 3)

#@staticmethod
def alignment_loss(
    pos_y, pos_x, batch,
):
    with torch.no_grad():
        num_nodes = batch.n_nodes
        total_nodes = pos_y.shape[0]
        num_graphs = batch.num_graphs
        pos_y_mean = global_mean_pool(pos_y, batch.batch)
        pos_x_mean = global_mean_pool(pos_x, batch.batch)
        y = pos_y - torch.repeat_interleave(pos_y_mean, num_nodes, dim=0)
        x = pos_x - torch.repeat_interleave(pos_x_mean, num_nodes, dim=0)
        a = y + x
        b = y - x
        a = a.view(total_nodes, 1, 3)
        b = b.view(total_nodes, 3, 1)
        tmp0 = torch.cat(
            [b.new_zeros((1, 1, 1)).expand(total_nodes, -1, -1), -b.permute(0, 2, 1)], dim=-1
        )
        eye = torch.eye(3).to(a).unsqueeze(0).expand(total_nodes, -1, -1)
        a = a.expand(-1, 3, -1)
        tmp1 = torch.cross(eye, a, dim=-1)
        tmp1 = torch.cat([b, tmp1], dim=-1)
        tmp = torch.cat([tmp0, tmp1], dim=1)
        tmpb = torch.bmm(tmp.permute(0, 2, 1), tmp).view(total_nodes, -1)
        tmpb = global_mean_pool(tmpb, batch.batch).view(num_graphs, 4, 4)
        w, v = torch.linalg.eigh(tmpb)
        min_rmsd = w[:, 0]
        min_q = v[:, :, 0]
        rotation = quaternion_to_rotation_matrix(min_q)
        t = pos_y_mean - torch.einsum("kj,kij->ki", pos_x_mean, rotation)
        rotation = torch.repeat_interleave(rotation, num_nodes, dim=0)
        t = torch.repeat_interleave(t, num_nodes, dim=0)
    pos_x = torch.einsum("kj,kij->ki", pos_x, rotation) + t
    loss = global_mean_pool((pos_y - pos_x).norm(dim=-1, keepdim=True), batch.batch).mean()
    return loss

def logger_itc(pl_module, loss_itc):  # 记录 ITC（Image-Text Co-Training）任务的损失，并返回损失值以供进一步处理。

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_itc_loss")(loss_itc)
    pl_module.log(f"itc/{phase}/loss", loss)
    return loss


def compute_mae(target, pred, mask):  # 计算了预测值和目标值之间的均方误差（MAE）。
    # mean = target.mean(dim=-1, keepdim=True)
    # var = target.var(dim=-1, keepdim=True)
    # target = (target - mean) / (var + 1.e-6) ** .5

    loss = (pred - target) ** 2  # 计算预测值和目标值之间的平方差。
    mean_loss = loss.mean(dim=-1)  # [N, L], mean loss per patch对每个样本的每个特征维度求均值，得到每个样本的平均损失。
    loss = (
                   mean_loss * mask).sum() / mask.sum()  # mean loss on removed patches使用掩码 mask 将每个样本的平均损失加权求和，并除以掩码中非零值的数量，得到整体的平均损失。
    return loss


def logger_mpp(pl_module, batch, data):
    '''

    这个 logger_mpp 函数用于记录模型在生成图像时的性能。它的主要步骤如下：

    使用模型进行推断，生成预测的图像。
    计算预测图像与目标图像之间的均方误差，这里使用了 compute_mae 函数。
    根据训练或验证阶段将损失记录到日志中。
    这个函数的目的是评估模型在生成图像任务上的表现，并将损失记录到日志中以进行后续的分析和监控。
    '''
    images, _, smiles, *el = batch
    infer_mpp = pl_module.infer(images, smiles, data= data, mask_image=True)
    target = infer_mpp["imgs"]
    pred = infer_mpp["pred"]
    mask = infer_mpp["mask"]
    loss_mpp = compute_mae(target, pred, mask)

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mpp_loss")(loss_mpp)
    pl_module.log(f"mpp/{phase}/loss", loss)

    return {
        "loss_mpp": loss_mpp,
    }


def logger_fpp(pl_module, batch, data):
    images, _, smiles, _, k_100, k_1000, k_10000,_,_ = batch

    infer_fpp = pl_module.infer(images, smiles, data= data)
    # fpp_labels = torch.tensor([list(map(int, list(x))) for x in fingerprint]).to(images.device)
    # fpp_logits_image = pl_module.fpp_score_images(infer_fpp['cls_feats_image'])
    # loss_fpp_image = pl_module.focal_loss(
    #     fpp_logits_image.view(-1, 100),
    #     k_100.view(-1),
    # )
    # fpp_logits_smiles = pl_module.fpp_score_smiles(infer_fpp['cls_feats_smiles'])
    # loss_fpp_smiles = F.cross_entropy(
    #     fpp_logits_smiles.view(-1, 100),
    #     k_100.view(-1),
    # )
    # total_loss = loss_fpp_image + loss_fpp_smiles

    fpp_logits_image100 = pl_module.fpp_score_images100(infer_fpp['cls_feats_image'])
    loss_fpp_image100 = pl_module.focal_loss(
        fpp_logits_image100.view(-1, 100),
        k_100.view(-1),
    )
    fpp_logits_smiles100 = pl_module.fpp_score_smiles100(infer_fpp['cls_feats_smiles'])
    loss_fpp_smiles100 = F.cross_entropy(
        fpp_logits_smiles100.view(-1, 100),
        k_100.view(-1),
    )

    fpp_logits_image1000 = pl_module.fpp_score_images1000(infer_fpp['cls_feats_image'])
    loss_fpp_image1000 = pl_module.focal_loss(
        fpp_logits_image1000.view(-1, 1000),
        k_1000.view(-1),
    )
    fpp_logits_smiles1000 = pl_module.fpp_score_smiles1000(infer_fpp['cls_feats_smiles'])
    loss_fpp_smiles1000 = F.cross_entropy(
        fpp_logits_smiles1000.view(-1, 1000),
        k_1000.view(-1),
    )

    fpp_logits_image10000 = pl_module.fpp_score_images10000(infer_fpp['cls_feats_image'])
    loss_fpp_image10000 = pl_module.focal_loss(
        fpp_logits_image10000.view(-1, 10000),
        k_10000.view(-1),
    )
    fpp_logits_smiles10000 = pl_module.fpp_score_smiles10000(infer_fpp['cls_feats_smiles'])
    loss_fpp_smiles10000 = F.cross_entropy(
        fpp_logits_smiles10000.view(-1, 10000),
        k_10000.view(-1),
    )

    total_loss = loss_fpp_image100 + loss_fpp_smiles100 + loss_fpp_image1000 + loss_fpp_smiles1000 + loss_fpp_image10000 + loss_fpp_smiles10000

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_fpp_loss")(total_loss)

    acc_image100 = getattr(pl_module, f"{phase}_fpp_accuracy")(
        fpp_logits_image100, k_100, k=True,
    )
    acc_smiles100 = getattr(pl_module, f"{phase}_fpp_accuracy")(
        fpp_logits_smiles100, k_100, k=True,
    )
    acc_image1000 = getattr(pl_module, f"{phase}_fpp_accuracy")(
        fpp_logits_image1000, k_1000, k=True,
    )
    acc_smiles1000 = getattr(pl_module, f"{phase}_fpp_accuracy")(
        fpp_logits_smiles1000, k_1000, k=True,
    )
    acc_image10000 = getattr(pl_module, f"{phase}_fpp_accuracy")(
        fpp_logits_image10000, k_10000, k=True,
    )
    acc_smiles10000 = getattr(pl_module, f"{phase}_fpp_accuracy")(
        fpp_logits_smiles10000, k_10000, k=True,
    )
    pl_module.log(f"fpp/{phase}/loss", loss)
    pl_module.log(f"fpp/{phase}/accuracy", (
            acc_image100 + acc_smiles100 + acc_image1000 + acc_smiles1000 + acc_image10000 + acc_smiles10000) / 6)
    print(f'acc_image100: {acc_image100},acc_smiles100:{acc_smiles100}, acc_image1000:{acc_image1000} , '
          f'acc_smiles1000:{acc_smiles1000}, acc_image10000:{acc_image10000}, acc_smiles10000:{acc_smiles10000}')
    return {
        "loss_fpp": total_loss,
        "acc_fpp_smiles100": acc_smiles100,
        "acc_fpp_image100": acc_image100,
        "acc_fpp_smiles1000": acc_smiles1000,
        "acc_fpp_image1000": acc_image1000,
        "acc_fpp_smiles10000": acc_smiles10000,
        "acc_fpp_image10000": acc_image10000,
    }


# def compute_MoleculeNet_classify(pl_module, batch):
#     smiles,images,label = batch
#     infer_MoleculeNet_BBBP = pl_module.infer(images, smiles)
#
#     MoleculeNet_BBBP_logits = pl_module.MoleculeNet_BBBP_score(infer_MoleculeNet_BBBP['cls_feats'])
#     loss_MoleculeNet_BBBP = pl_module.focal_loss(
#         MoleculeNet_BBBP_logits.view(-1, 2),
#         label.view(-1),
#     )
#     phase = "train" if pl_module.training else "val"
#     loss = getattr(pl_module, f"{phase}_MoleculeNet_BBBP_loss")(loss_MoleculeNet_BBBP)
#     acc = getattr(pl_module, f"{phase}_MoleculeNet_BBBP_accuracy")(
#         MoleculeNet_BBBP_logits, label
#     )
#     auroc = getattr(pl_module,f"{phase}_MoleculeNet_BBBP_auroc").compute(
#         MoleculeNet_BBBP_logits, label
#     )
#     pl_module.log(f"MoleculeNet_BBBP/{phase}/loss", loss)
#     pl_module.log(f"MoleculeNet_BBBP/{phase}/accuracy", acc)
#     pl_module.log(f"MoleculeNet_BBBP/{phase}/auroc",auroc)
#     return {
#         "loss_MoleculeNet_BBBP": loss_MoleculeNet_BBBP,
#         "acc_MoleculeNet_BBBP": acc,
#         'auroc': auroc,
#     }

def compute_MoleculeNet_classify(pl_module, batch, data, testing=False):
    '''
    compute_MoleculeNet_classify 函数的主要功能是计算并记录模型在 MoleculeNet 分类任务上的性能指标，包括损失、准确率和 AUROC（Area Under the Receiver Operating Characteristic Curve）。
    '''
    smiles, images, label,pos = batch
    infer_MoleculeNet_classify = pl_module.infer(images, smiles, data= data, mode="raw")
    MoleculeNet_classify_logits = pl_module.MoleculeNet_classify_score(infer_MoleculeNet_classify['cls_feats_all'])

    loss_MoleculeNet_classify = pl_module.focal_loss(
        MoleculeNet_classify_logits.view(-1, 2),
        label.view(-1),
    )
    phase = "train" if pl_module.training else "val"
    if testing == True:
        phase = "test"
    loss = getattr(pl_module, f"{phase}_MoleculeNet_classify_loss")(loss_MoleculeNet_classify)
    acc = getattr(pl_module, f"{phase}_MoleculeNet_classify_accuracy")(
        MoleculeNet_classify_logits, label, k=True,
    )
    #print(MoleculeNet_classify_logits.shape)
    #print(type(MoleculeNet_classify_logits))
    auroc = getattr(pl_module, f"{phase}_MoleculeNet_classify_auroc")(
        MoleculeNet_classify_logits, label
    )
    precision = getattr(pl_module, f"{phase}_MoleculeNet_classify_precision")(MoleculeNet_classify_logits, label)
    recall = getattr(pl_module, f"{phase}_MoleculeNet_classify_recall")(MoleculeNet_classify_logits, label)
    f1 = getattr(pl_module, f"{phase}_MoleculeNet_classify_f1")(MoleculeNet_classify_logits, label)
    pl_module.log(f"MoleculeNet_classify/{phase}/loss", loss)
    pl_module.log(f"MoleculeNet_classify/{phase}/accuracy", acc)
    pl_module.log(f"MoleculeNet_classify/{phase}/auroc", auroc)
    pl_module.log(f"MoleculeNet_classify/{phase}/precision", precision)
    pl_module.log(f"MoleculeNet_classify/{phase}/recall", recall)
    pl_module.log(f"MoleculeNet_classify/{phase}/f1", f1)

    return {
        "loss_MoleculeNet_classify": loss_MoleculeNet_classify,
        "acc_MoleculeNet_classify": acc,
        'auroc': auroc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        'logits': MoleculeNet_classify_logits,
    }


'''
def compute_MoleculeNet_classify(pl_module, batch,testing=False):

    #compute_MoleculeNet_classify 函数的主要功能是计算并记录模型在 MoleculeNet 分类任务上的性能指标，包括损失、准确率和 AUROC（Area Under the Receiver Operating Characteristic Curve）。

    smiles,images,label = batch
    infer_MoleculeNet_classify = pl_module.infer(images, smiles)
    MoleculeNet_classify_logits = pl_module.MoleculeNet_classify_score(infer_MoleculeNet_classify['cls_feats'])
    loss_MoleculeNet_classify = pl_module.focal_loss(
        MoleculeNet_classify_logits.view(-1, 2),
        label.view(-1),
    )
    phase = "train" if pl_module.training else "val"
    if testing == True:
        phase = "test"
    loss = getattr(pl_module, f"{phase}_MoleculeNet_classify_loss")(loss_MoleculeNet_classify)
    acc = getattr(pl_module, f"{phase}_MoleculeNet_classify_accuracy")(
        MoleculeNet_classify_logits, label,k = True,
    )
    auroc = getattr(pl_module,f"{phase}_MoleculeNet_classify_auroc")(
        MoleculeNet_classify_logits, label
    )
    pl_module.log(f"MoleculeNet_classify/{phase}/loss", loss)
    pl_module.log(f"MoleculeNet_classify/{phase}/accuracy", acc)
    pl_module.log(f"MoleculeNet_classify/{phase}/auroc",auroc)

    return {
        "loss_MoleculeNet_classify": loss_MoleculeNet_classify,
        "acc_MoleculeNet_classify": acc,
        'auroc': auroc,
        'logits': MoleculeNet_classify_logits,
    }
'''


def imageMol_classify(pl_module, batch, testing=False):
    smiles, images, label = batch
    infer_MoleculeNet_classify = pl_module.infer(images, smiles)
    MoleculeNet_classify_logits = infer_MoleculeNet_classify['logits']
    loss_MoleculeNet_classify = pl_module.focal_loss(
        MoleculeNet_classify_logits.view(-1, 2),
        label.view(-1),
    )
    phase = "train" if pl_module.training else "val"
    if testing == True:
        phase = "test"
    loss = getattr(pl_module, f"{phase}_MoleculeNet_classify_loss")(loss_MoleculeNet_classify)
    acc = getattr(pl_module, f"{phase}_MoleculeNet_classify_accuracy")(
        MoleculeNet_classify_logits, label, k=True,
    )
    auroc = getattr(pl_module, f"{phase}_MoleculeNet_classify_auroc")(
        MoleculeNet_classify_logits, label
    )
    pl_module.log(f"MoleculeNet_classify/{phase}/loss", loss)
    pl_module.log(f"MoleculeNet_classify/{phase}/accuracy", acc)
    pl_module.log(f"MoleculeNet_classify/{phase}/auroc", auroc)

    return {
        "loss_MoleculeNet_classify": loss_MoleculeNet_classify,
        "acc_MoleculeNet_classify": acc,
        'auroc': auroc,
        'logits': MoleculeNet_classify_logits,
    }


def chemberta_classify(pl_module, batch, testing=False):
    smiles, images, label = batch
    infer_MoleculeNet_classify = pl_module.infer(images, smiles)
    MoleculeNet_classify_logits = pl_module.MoleculeNet_classify_score(infer_MoleculeNet_classify['smiles_feats'])
    loss_MoleculeNet_classify = pl_module.focal_loss(
        MoleculeNet_classify_logits.view(-1, 2),
        label.view(-1),
    )
    phase = "train" if pl_module.training else "val"
    if testing == True:
        phase = "test"
    loss = getattr(pl_module, f"{phase}_MoleculeNet_classify_loss")(loss_MoleculeNet_classify)
    acc = getattr(pl_module, f"{phase}_MoleculeNet_classify_accuracy")(
        MoleculeNet_classify_logits, label, k=True,
    )
    auroc = getattr(pl_module, f"{phase}_MoleculeNet_classify_auroc")(
        MoleculeNet_classify_logits, label
    )
    pl_module.log(f"MoleculeNet_classify/{phase}/loss", loss)
    pl_module.log(f"MoleculeNet_classify/{phase}/accuracy", acc)
    pl_module.log(f"MoleculeNet_classify/{phase}/auroc", auroc)

    return {
        "loss_MoleculeNet_classify": loss_MoleculeNet_classify,
        "acc_MoleculeNet_classify": acc,
        'auroc': auroc,
        'logits': MoleculeNet_classify_logits,
    }


def visualT_classify(pl_module, batch, testing=False):
    '''
    visualT_classify 函数与之前的 compute_MoleculeNet_classify 函数类似，也是用于计算并记录模型在 MoleculeNet 分类任务上的性能指标。但是，它使用的输入特征是图像特征而不是语义特征。
    '''
    smiles, images, label = batch
    infer_MoleculeNet_classify = pl_module.infer(images, smiles)
    MoleculeNet_classify_logits = pl_module.MoleculeNet_classify_score(infer_MoleculeNet_classify['image_feats'])
    loss_MoleculeNet_classify = pl_module.focal_loss(
        MoleculeNet_classify_logits.view(-1, 2),
        label.view(-1),
    )
    phase = "train" if pl_module.training else "val"
    if testing == True:
        phase = "test"
    loss = getattr(pl_module, f"{phase}_MoleculeNet_classify_loss")(loss_MoleculeNet_classify)
    acc = getattr(pl_module, f"{phase}_MoleculeNet_classify_accuracy")(
        MoleculeNet_classify_logits, label, k=True,
    )
    auroc = getattr(pl_module, f"{phase}_MoleculeNet_classify_auroc")(
        MoleculeNet_classify_logits, label
    )
    pl_module.log(f"MoleculeNet_classify/{phase}/loss", loss)
    pl_module.log(f"MoleculeNet_classify/{phase}/accuracy", acc)
    pl_module.log(f"MoleculeNet_classify/{phase}/auroc", auroc)

    return {
        "loss_MoleculeNet_classify": loss_MoleculeNet_classify,
        "acc_MoleculeNet_classify": acc,
        'auroc': auroc,
        'logits': MoleculeNet_classify_logits,
    }


#def compute_MoleculeNet_regress(pl_module, batch, data, testing=False):
#    smiles, images, label,pos = batch
#    label = label.float()  # 将标签转换为 Float 类型
#
#    infer_MoleculeNet_regress = pl_module.infer(images, smiles, data= data, mode="raw")
#
#    MoleculeNet_regress_logits = pl_module.MoleculeNet_regress_score(infer_MoleculeNet_regress['cls_feats_all'])
#    # rmse = RMSE(images.device)
#    criterion = nn.MSELoss(reduction="mean")
#    s = criterion(
#        MoleculeNet_regress_logits.view(-1),
#        label.view(-1)
#    )
#    print(f's:{s}')
#    loss_MoleculeNet_regress = torch.sqrt(s)
#    print(f"Loss (RMSE): {loss_MoleculeNet_regress.item():.4f}")
#    phase = "train" if pl_module.training else "val"
#    if testing == True:
#        phase = "test"
#    # 保存损失状态
#    loss = getattr(pl_module, f"{phase}_MoleculeNet_regress_loss")(loss_MoleculeNet_regress)
#    pl_module.log(f"MoleculeNet_regress/{phase}/loss", loss)
#    print(loss)
#
#    return {
#        "loss_MoleculeNet_regress": loss_MoleculeNet_regress,
#    }
from scipy.stats import pearsonr, spearmanr
import csv
# 在函数外部定义全局变量来累积数据
smiles_list = []
logits_list = []
def compute_MoleculeNet_regress(pl_module, batch, data, testing=False):
    smiles, images, label,pos = batch
    #smiles = [s.replace(r'\/', '/') for s in smiles]  # 移除 \/ 中的反斜杠

    label = label.float()  # 将标签转换为 Float 类型
    #label = label.long()

    infer_MoleculeNet_regress = pl_module.infer(images, smiles, data= data, mode="raw")

    MoleculeNet_regress_logits = pl_module.MoleculeNet_regress_score(infer_MoleculeNet_regress['cls_feats_all'])


   #     # 保存 smiles 和 logits 到本地文件
   # with open('/root/CXX/my/myCLIP/FDA_predict.csv', mode='a', newline='') as file:
   #     writer = csv.writer(file)
   #     for s, l in zip(smiles, MoleculeNet_regress_logits.detach().cpu().numpy().flatten()):
   #         writer.writerow([s, l])
    
    #print(label,MoleculeNet_regress_logits)
    # rmse = RMSE(images.device)
    criterion = nn.MSELoss(reduction="mean")
    s = criterion(
        MoleculeNet_regress_logits.view(-1),
        label.view(-1)
    )
    #print(f's:{s}')
    loss_MoleculeNet_regress = s.pow(0.5)
    #print(f"Loss (RMSE): {loss_MoleculeNet_regress.item():.4f}")


    # 计算皮尔逊相关系数
    pearson_r = pearsonr(
        MoleculeNet_regress_logits.detach().cpu().numpy().flatten(),  # 模型预测值
        label.detach().cpu().numpy().flatten()  # 真实标签
    )[0]  # pearsonr 返回 (r, p-value)，我们只需要 r
    print(f'pearson_r',pearson_r)

    # 计算 Spearman's r
    spearman_r = spearmanr(
        MoleculeNet_regress_logits.detach().cpu().numpy().flatten(),  # 模型预测值
        label.detach().cpu().numpy().flatten()  # 真实标签
    )[0]  # spearmanr 返回 (r, p-value)，我们只需要 r
    print(f'Spearman\'s r: {spearman_r}')

    print(f'pl_module.training:',pl_module.training)
    phase = "train" if pl_module.training else "val"
    if testing == True:
        phase = "test"
    # 保存损失状态
    loss = getattr(pl_module, f"{phase}_MoleculeNet_regress_loss")(loss_MoleculeNet_regress)

    pl_module.log(f"MoleculeNet_regress/{phase}/loss", loss)

    

    

    return {
        "loss_MoleculeNet_regress": loss_MoleculeNet_regress,
        "pearson_r": pearson_r,  # 返回皮尔逊相关系数
        "spearman_r": spearman_r,  # 返回 Spearman's r
    }




def compute_ocsr_finturn(pl_module, batch):  # 这个函数用于计算和记录在 OCSR（One-Class Soft Rejection）任务上的性能指标，通常用于对抗训练或者迁移学习中。
    images, images_false, smiles, image_id = batch

    infer_ocsr = pl_module.infer(images, smiles, images_false, change_smi=True)
    ocsr_logits = pl_module.itm_score(infer_ocsr["cls_feats"])
    ocsr_labels = infer_ocsr['ocsr_labels'].to(images.device)
    loss_ocsr = pl_module.focal_loss(
        ocsr_logits,
        ocsr_labels.long()
    )
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_ocsr_finturn_loss")(loss_ocsr)  # Metrics.forwrad()
    acc = getattr(pl_module, f"{phase}_ocsr_finturn_accuracy")(
        ocsr_logits, ocsr_labels.long()
    )
    pl_module.log(f"ocsr_finturn/{phase}/loss", loss)
    pl_module.log(f"ocsr_finturn/{phase}/accuracy", acc)
    return {
        "loss_itm": loss_ocsr,
        "acc_itm": acc,
    }


def compute_ocsr(pl_module, batch):  # 计算 OCSR（One-Class Soft Rejection）任务的输出，它不仅返回了模型的预测结果，还将图像的 ID 与预测结果一起返回。具体步骤如下：
    smiles, image, image_id = batch
    infer_itm = pl_module.infer(image, smiles)
    logits = pl_module.itm_score(infer_itm["cls_feats"])
    predictions = logits.argmax(dim=-1)
    print(predictions)
    image_id = torch.tensor(image_id)
    result = torch.stack((image_id, predictions), dim=1)
    return {
        'result': result,
    }


def Uamp(pl_module, batch):
    smiles, images, label = batch
    infer_MoleculeNet = pl_module.infer(images, smiles)
    cls_feats = infer_MoleculeNet['cls_feats']
    cls_feats_image = infer_MoleculeNet['cls_feats_image']
    cls_feats_smiles = infer_MoleculeNet['cls_feats_smiles']
    return {
        'cls_feats': cls_feats,
        'cls_feats_image': cls_feats_image,
        'cls_feats_smiles': cls_feats_smiles,
        'labels': label,
        'smiles': smiles
    }


def Uamp(pl_module, batch):
    smiles, images, label = batch
    infer_MoleculeNet = pl_module.infer(images, smiles)
    cls_feats = infer_MoleculeNet['cls_feats']
    cls_feats_image = infer_MoleculeNet['cls_feats_image']
    cls_feats_smiles = infer_MoleculeNet['cls_feats_smiles']
    return {
        'cls_feats': cls_feats,
        'cls_feats_image': cls_feats_image,
        'cls_feats_smiles': cls_feats_smiles,
        'labels': label,
        'smiles': smiles
    }


def Uamp_imageMol(pl_module, batch):
    smiles, images, label = batch
    infer_MoleculeNet = pl_module.infer(images, smiles)
    embedding = infer_MoleculeNet['embedding'].squeeze(-1).squeeze(-1)
    return {
        'embedding': embedding,
        'labels': label,
        'smiles': smiles
    }


def write_auroc(pl_module, batch):
    smiles, images, label = batch
    infer_MoleculeNet_classify = pl_module.infer(images, smiles)
    MoleculeNet_classify_logits = pl_module.MoleculeNet_classify_score(infer_MoleculeNet_classify['cls_feats'])

    return {
        'labels': label,
        'logits': MoleculeNet_classify_logits,
    }


class FocalLoss(nn.Module):  # Focal Loss，这是交叉熵损失的一个变种，通常用于具有不平衡类分布的分类任务。它通过降低对分类良好的示例的权重来考虑类不平衡。

    def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import numpy as np

import torch


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):  # 生成二维网格的正弦和余弦位置嵌入（positional embedding）。
    """

    embed_dim：嵌入维度，即每个位置的嵌入向量的长度。
    grid_size：网格的高度和宽度，即二维网格的尺寸。
    cls_token：一个布尔值，指示是否包含额外的分类标记（cls token）。
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):  # 生成一维正弦和余弦位置嵌入。
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


def cost_matrix_cosine(x, y, eps=1e-5):
    """Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]"""
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    cosine_dist = 1 - cosine_sim
    return cosine_dist


def trace(x):
    """ compute trace of input tensor (batched) """
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.bool, device=x.device).unsqueeze(0).expand_as(x)
    trace = x.masked_select(mask).contiguous().view(b, n).sum(dim=-1, keepdim=False)
    return trace


@torch.no_grad()
def ipot(C, x_len, x_pad, y_len, y_pad, joint_pad, beta, iteration, k):
    """ [B, M, N], [B], [B, M], [B], [B, N], [B, M, N]"""
    b, m, n = C.size()
    sigma = torch.ones(b, m, dtype=C.dtype, device=C.device) / x_len.unsqueeze(1)
    T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
    A = torch.exp(-C.transpose(1, 2) / beta)

    # mask padded positions
    sigma.masked_fill_(x_pad, 0)
    joint_pad = joint_pad.transpose(1, 2)
    T.masked_fill_(joint_pad, 0)
    A.masked_fill_(joint_pad, 0)

    # broadcastable lengths
    x_len = x_len.unsqueeze(1).unsqueeze(2)
    y_len = y_len.unsqueeze(1).unsqueeze(2)

    # mask to zero out padding in delta and sigma
    x_mask = (x_pad.to(C.dtype) * 1e4).unsqueeze(1)
    y_mask = (y_pad.to(C.dtype) * 1e4).unsqueeze(1)

    for _ in range(iteration):
        Q = A * T  # bs * n * m
        sigma = sigma.view(b, m, 1)
        for _ in range(k):
            delta = 1 / (y_len * Q.matmul(sigma).view(b, 1, n) + y_mask)
            sigma = 1 / (x_len * delta.matmul(Q) + x_mask)
        T = delta.view(b, n, 1) * Q * sigma
    T.masked_fill_(joint_pad, 0)
    return T
