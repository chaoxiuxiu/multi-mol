import os
import warnings
import random
import numpy as np
import torch
from torchvision.transforms import transforms
from rdkit import Chem
from rdkit.Chem import AllChem
from evaluate import metrics_utils
from models import predictor
from models import utils
import time


from models.utils_mol_KANO1 import logger_itm, logger_mlm, logger_gcl, FocalLoss, logger_mpp, \
    logger_fpp, logger_gtm, mask2d, conf2mol, mol2conf, \
    compute_MoleculeNet_classify, compute_MoleculeNet_regress, compute_ocsr, compute_ocsr_finturn, Uamp, write_auroc
from models.vit import createVisualModel
from models.mae import mae_vit_base_patch16_dec512d8b
from models.model import BridgeTowerBlock, GNBlock, AtomEmbeddingwithMask, PosEmbeddingwithMask, MolDecoder
from transformers.models.bert.modeling_bert import BertConfig
from models.med import BertCrossLayer
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from transformers import AutoModelForMaskedLM, AutoTokenizer
from torch_sparse import SparseTensor
import re
# from .evaluate import evaluate, evaluate_predictions
# from .predict import predict
# from .train import train
# from chemprop.data import StandardScaler
from chemprop1.data.utils import get_class_sizes, get_data_from_smiles, get_task_names, split_data, get_data
from chemprop1.models import build_model, build_pretrain_model, add_functional_prompt, MoleculeModel
from chemprop1.nn_utils import param_count
from chemprop1.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint, \
    makedirs, save_checkpoint
from chemprop1.data import MoleculeDataset
from tqdm import tqdm, trange
from chemprop1.models import ContrastiveLoss
from chemprop1.torchlight import initialize_exp, snapshot
from logging import Logger
from torch_geometric.data import Data
from pretrain3d.utils.graph import smiles2graphwithface
from pretrain3d.utils.gt import isomorphic_core
from copy import deepcopy
import ast
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from pretrain3d.utils.features import get_atom_feature_dims, get_bond_feature_dims
from pretrain3d.model.conv import (
    MLP,
    DropoutIfTraining,
    MetaLayer,
    MLPwoLastAct,
    MLPwithselfBN,
    MLPwoLastActwithselfBN,
)
import torch.nn.functional as F
from pretrain3d.utils.torch_util import GradMultiply
import torch
from torch_geometric.data import Batch
from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.nn import global_mean_pool

# import wandb
warnings.filterwarnings("ignore")
# wandb.init(project="myCLIP", entity="zhxiang",settings=wandb.Settings(start_method="fork"))
# 模型

from argparse import Namespace
from rdkit import RDLogger

_REDUCER_NAMES = {"sum": global_add_pool, "mean": global_mean_pool, "max": global_max_pool}
# 配置 RDKit 的日志级别
RDLogger.DisableLog('rdApp.*')


torch.set_grad_enabled(True)

# 从给定信息中创建参数对象
args_dict = {
    'num_gpus': [1],
    'data_path': './data/qm7.csv',
    'use_compound_names': False,
    'max_data_size': None,
    'test': False,
    'features_only': False,
    'features_generator': None,
    'features_path': None,
    'save_dir': 'dumped/0516-finetune/qm7/run_0',
    'save_smiles_splits': False,
    'checkpoint_dir': None,
    'graph_checkpoint_path': 'E:/autodl-tmp/KANO/dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl',
    'dataset_type': 'regression',
    'multiclass_num_classes': 3,
    'separate_val_path': None,
    'separate_val_features_path': None,
    'separate_test_path': None,
    'separate_test_features_path': None,
    'split_type': 'scaffold_balanced',
    'split_sizes': [0.8, 0.1, 0.1],
    'num_runs': 20,
    'folds_file': None,
    'val_fold_index': None,
    'test_fold_index': None,
    'crossval_index_dir': None,
    'crossval_index_file': None,
    'seed': 43,
    'metric': 'mae',
    'quiet': False,
    'log_frequency': 10,
    'show_individual_scores': False,
    'no_cache': False,
    'config_path': None,
    'epochs': 100,
    'batch_size': 12,  # 256
    'warmup_epochs': 2.0,
    'init_lr': 0.0001,
    'max_lr': 0.001,
    'final_lr': 0.0001,
    'temperature': 0.1,
    'encoder_name': 'CMPNN',
    'ensemble_size': 1,
    'hidden_size': 300,  # 300
    'bias': False,
    'depth': 3,
    'dropout': 0.0,
    'activation': 'ReLU',
    'undirected': False,
    'ffn_hidden_size': 300,  # 300
    'ffn_num_layers': 2,
    'atom_messages': False,
    'dump_path': 'dumped',
    'exp_name': 'finetune',
    'exp_id': 'qm7',
    'step': 'pretrain',  # 微调：functional_prompt，预训练：pretrain,finetune_add,finetune_concat
    'cuda': True,  # 这里将 cuda 设置为 True
    'features_scaling': True,
    'minimize_score': True,
    'checkpoint_paths': ['./dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl'],
    'use_input_features': None,
    'num_lrs': 1,
    'task_names': ['p_np'],
    'num_tasks': 1,
    'features_size': None,
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',  # add
    "global_reducer": "sum",
    "node_reducer": "sum",
    "graph_pooling": "sum",
    "dropedge_rate": 0.1,
    "dropnode_rate": 0.1,
    "num_layers": 12,
    "latent_size": 768,
    "mlp_hidden_size": 512,
    "mlp_layers": 2,
    "use_layer_norm": False,
    "log_interval": 100,
    "encoder_dropout": 0.0,
    "pooler_dropout": 0.0,
    "layernorm_before": False,
    "use_bn": True,  # False
    "weight_decay": 1e-2,
    "beta2": 0.999,
    "period": 10,
    "enable_tb": False,
    "train_subset": False,
    "use_face": False,
    "global_attn": False,
    "node_attn": True,
    "face_attn": False,
    "grad_norm": None,
    "attr_predict": False,
    "random_rotation": False,
    "ap_hid_size": None,
    "ap_mlp_layers": None,
    "local_rank": 0,
    "pred_pos_residual": True,
    "raw_with_pos": False,
    "eval_from": None,
    "mask_prob": 0.15,
    "pos_mask_prob": None,
    "tasks": ["mask", "mol2conf", "conf2mol"],
    "restore": False,
}

# 创建命名空间参数对象
args = Namespace(**args_dict)

print("NCCL_TIMEOUT_MS:", os.environ.get("NCCL_TIMEOUT_MS"))


# Step 1: 定义计算 OT 的函数
def compute_ot(smiles_feats, other_feats, reg=1e-3):
    # 计算两组特征之间的欧氏距离矩阵
    distance_matrix = ot.dist(smiles_feats.cpu().detach().numpy(),
                              other_feats.cpu().detach().numpy(),
                              metric='euclidean')

    # 定义均匀的权重
    bs = smiles_feats.shape[0]
    smiles_weights = torch.ones((bs,)) / bs
    other_weights = torch.ones((bs,)) / bs

    # 计算最优传输矩阵
    ot_matrix = ot.sinkhorn(smiles_weights.cpu().detach().numpy(),
                            other_weights.cpu().detach().numpy(),
                            distance_matrix,
                            reg)

    # 转换为 PyTorch tensor 并返回
    return torch.tensor(ot_matrix, dtype=torch.float32, device=smiles_feats.device)


import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()

        # Linear layers for query, key, and value projections
        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)

        # Linear layer to combine attention output
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)

        # Linear projections for Q, K, and V
        Q = self.query_linear(x)  # (batch_size, seq_len, embed_dim)
        K = self.key_linear(x)  # (batch_size, seq_len, embed_dim)
        V = self.value_linear(x)  # (batch_size, seq_len, embed_dim)

        # Compute attention scores (Q * K^T) / sqrt(embed_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(Q.size(-1), dtype=torch.float32))  # (batch_size, seq_len, seq_len)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, seq_len, seq_len)

        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, seq_len, embed_dim)

        # Pass through final linear layer
        output = self.out_linear(attn_output)  # (batch_size, seq_len, embed_dim)

        return output


# print(args.latent_size)
class myCLIP(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        # image像素

        self.config = config
        #  wandb.config = config
        # 这段代码是在使用 PyTorch 分布式训练时的初始化操作，主要包括两个步骤：
        if torch.distributed.is_initialized():
            # 主线程
            if torch.distributed.get_rank() == 0:  # 如果当前环境已经初始化了分布式训练（通过 torch.distributed.is_initialized() 判断）
                # image encoder initialize
                createVisualModel(config['image_size'])  # 调用 createVisualModel(config['image_size']) 来初始化图像编码器。
                # Smiles encoder initialize
                AutoModelForMaskedLM.from_pretrained(
                    "E:/autodl-tmp/myCLIP/models/DeepChem/ChemBERTa-77M-MLM")  # 调用 AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MLM") 来初始化 Smiles 编码器。这里使用了预训练的模型 "DeepChem/ChemBERTa-77M-MLM"。
            torch.distributed.barrier()  # 调用 torch.distributed.barrier()，这个函数会阻塞进程，直到所有进程都达到了这个同步点。这样做是为了确保所有进程都在这个点上同步了。
        # vit,已 load_state_dict
        # self.visual_encoder = createVisualModel()
        self.visual_encoder = mae_vit_base_patch16_dec512d8b()  # 初始化图像编码器 self.visual_encoder，这里使用了一个预定义的视觉编码器。

        # self.tokenizer = self.init_tokenizer()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "E:/autodl-tmp/myCLIP/models/DeepChem/ChemBERTa-77M-MLM")
        self.vocab = self.tokenizer.vocab
        # smiles encoder initialize
        # 初始化分子序列（SMILES）的 tokenizer 和编码器。使用 AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MLM") 初始化预训练的 SMILES 编码器，并调整 token embeddings 的大小为指定的 config['input_smiles_embed_size']。
        self.smiles_encoder = AutoModelForMaskedLM.from_pretrained(
            "E:/autodl-tmp/myCLIP/models/DeepChem/ChemBERTa-77M-MLM")
        self.smiles_encoder.resize_token_embeddings(config['input_smiles_embed_size'])

        self.graph_encoder = build_model(args, encoder_name=args.encoder_name)
        # self.graph_encoder.load_state_dict(torch.load(args.graph_checkpoint_path, map_location='cpu'), strict=False)
        self.graph_encoder2 = build_model(args, encoder_name=args.encoder_name)
        # self.graph_encoder2.load_state_dict(torch.load(args.graph_checkpoint_path, map_location='cpu'), strict=False)
        if args.step == 'functional_prompt':
            add_functional_prompt(self.graph_encoder, args)
            add_functional_prompt(self.graph_encoder2, args)

        # add 3d
        self.latent_size = args.latent_size
        self.encoder_edge = MLP(
            sum(get_bond_feature_dims()),
            [args.mlp_hidden_size] * args.mlp_layers + [args.latent_size],
            use_layer_norm=args.use_layer_norm,
        )
        self.node_embedding = AtomEmbeddingwithMask(
            latent_size=args.latent_size,
            mlp_hidden_size=args.mlp_hidden_size,
            mlp_layers=args.mlp_layers,
            use_layernorm=args.use_layer_norm,
            mask_prob=args.mask_prob,
        )
        self.global_init = nn.Parameter(torch.zeros((1, args.latent_size), dtype=torch.float32))
        self._init_weight()
        if args.use_face:
            self.encoder_face = MLP(
                args.latent_size * 3,
                [args.mlp_hidden_size] * args.mlp_layers + [args.latent_size],
                use_layer_norm=args.use_layer_norm,
            )
        else:
            self.encoder_face = None

        args.pos_mask_prob = args.pos_mask_prob if args.pos_mask_prob is not None else args.mask_prob
        pos_embedding = PosEmbeddingwithMask(args.latent_size, args.pos_mask_prob, raw_with_pos=args.raw_with_pos)
        args.num_message_passing_steps = args.num_layers
        args.face_reducer = args.node_reducer
        # args.pos_mask_prob = args.pos_mask_prob if args.pos_mask_prob is not None else args.mask_prob
        self.gnn_layers = GNBlock(
            mlp_hidden_size=args.mlp_hidden_size,  # （MLP）隐藏层的大小
            mlp_layers=args.mlp_layers,  # 隐藏层的数量。
            latent_size=args.latent_size,  # 潜在空间的维度
            use_layer_norm=args.use_layer_norm,  # 是否在MLP中使用层归一化
            num_message_passing_steps=args.num_message_passing_steps,  # 消息传递步骤的数量
            global_reducer=args.global_reducer,
            node_reducer=args.node_reducer,
            face_reducer=args.face_reducer,  # 分别用于全局、节点和面的特征聚合函数，如 "sum"、"mean" 或 "max"。
            dropedge_rate=args.dropedge_rate,
            dropnode_rate=args.dropnode_rate,
            use_face=args.use_face,
            dropout=args.dropout,
            layernorm_before=args.layernorm_before,
            encoder_dropout=args.encoder_dropout,
            use_bn=args.use_bn,
            global_attn=args.global_attn,
            node_attn=args.node_attn,
            face_attn=args.face_attn,  # 是否使用全局、节点或面的注意力机制。
            pred_pos_residual=args.pred_pos_residual,  # 是否预测位置的残差。
            pos_embedding=pos_embedding,
        )
        if args.attr_predict:  # 是否预测属性
            ap_hid_size = args.mlp_hidden_size if args.ap_hid_size is None else args.ap_hid_size
            ap_mlp_layers = args.mlp_layers if args.ap_mlp_layers is None else args.ap_mlp_layers
            self.attr_decoder = MLPwoLastAct(
                args.latent_size,
                [ap_hid_size] * ap_mlp_layers + [args.num_tasks],
                use_layer_norm=False,
                dropout=args.pooler_dropout,
                use_bn=args.use_bn,
            )
        else:
            self.attr_decoder = MolDecoder(
                latent_size=args.latent_size,
                hidden_size=args.mlp_hidden_size,
                mlp_layers=args.mlp_layers,
                use_bn=args.use_bn,
                pooler_dropout=args.pooler_dropout,
            )
        self.attr_predict = args.attr_predict
        self.pooling = _REDUCER_NAMES[args.graph_pooling]
        self.aggregate_edges_for_face_fn = _REDUCER_NAMES[args.face_reducer]
        self.use_face = args.use_face
        gradmultiply = 0.1  # 梯度乘法的系数，可能用于控制特定层的梯度流动。
        self.gradmultiply = gradmultiply

        # 初始化smiles proj层权重,初始化image proj权重
        # 初始化 SMILES 和图像之间的跨模态投影层。这里分别使用 nn.Linear 初始化了 self.cross_modal_smiles_transform 和 self.cross_modal_image_transform 两个线性层，并调用了 utils.init_weights 函数来初始化权重。
        self.cross_modal_smiles_transform = nn.Linear(config['input_smiles_embed_size'],
                                                      config['hidden_size'])  # config['input_smiles_embed_size']
        self.cross_modal_smiles_transform.apply(utils.init_weights)
        self.cross_modal_image_transform = nn.Linear(config['input_image_embed_size'], config['hidden_size'])
        self.cross_modal_image_transform.apply(utils.init_weights)
        self.cross_modal_graph_transform = nn.Linear(args.hidden_size, config['hidden_size'])
        self.cross_modal_graph_transform.apply(utils.init_weights)
        # self.cross_modal_pos_transform = nn.Linear(args.latent_size, config['hidden_size'])
        # self.cross_modal_pos_transform.apply(utils.init_weights)

        # 两种类型token
        # 初始化用于区分不同类型 token 的 token type embeddings。这里使用了 nn.Embedding 来初始化 self.token_type_embeddings，大小为 2（代表两种类型），并同样调用了 utils.init_weights 来初始化权重。
        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        # self.token_type_embeddings = nn.Embedding(3, config["hidden_size"])
        # self.token_type_embeddings = nn.Embedding(4, config["hidden_size"])
        self.token_type_embeddings.apply(utils.init_weights)

        # 这段代码配置了 BERT 编码器和解码器的参数，并初始化了跨模态的注意力层。具体步骤如下：

        # Bert encoder配置,配置 BERT 编码器的参数 bert_config_encoder。其中包括了词汇大小（vocab_size）、隐藏层大小（hidden_size）、编码器层数（num_hidden_layers）、注意力头数（num_attention_heads）等。
        bert_config_encoder = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_smiles_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
            is_decoder=False,
        )

        # cross-attention-encoder
        # 初始化跨模态注意力层。使用 nn.ModuleList 创建了长度为 config['num_top_layer'] 的模块列表，每个模块都是一个 BertCrossLayer，参数为 bert_config_encoder，并使用 utils.init_weights 函数初始化权重。
        self.cross_modal_image_layers = nn.ModuleList(
            [BertCrossLayer(bert_config_encoder) for _ in range(config['num_top_layer'])])
        self.cross_modal_image_layers.apply(utils.init_weights)
        self.cross_modal_smiles_layers = nn.ModuleList(
            [BertCrossLayer(bert_config_encoder) for _ in range(config['num_top_layer'])])
        self.cross_modal_smiles_layers.apply(utils.init_weights)
        self.cross_modal_graph_layers = nn.ModuleList(
            [BertCrossLayer(bert_config_encoder) for _ in range(config['num_top_layer'])])
        self.cross_modal_graph_layers.apply(utils.init_weights)
        self.cross_modal_pos_layers = nn.ModuleList(
            [BertCrossLayer(bert_config_encoder) for _ in range(config['num_top_layer'])])
        self.cross_modal_pos_layers.apply(utils.init_weights)

        # Bert config配置
        # 配置 BERT 解码器的参数 bert_config_decoder。与编码器类似，但这里将 is_decoder 参数设置为 True，表示这是一个解码器。
        bert_config_decoder = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_smiles_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
            is_decoder=True,
        )

        # 以上操作主要用于配置和初始化 BERT 模型的各个组件，为后续的训练和推理做准备。

        # # cross-attention-decoder
        # self.cross_modal_image_layers = nn.ModuleList(
        #     [BertCrossLayer(bert_config) for _ in range(config['num_end_layer'])])
        # self.cross_modal_image_layers.apply(utils.init_weights)
        # self.cross_modal_smiles_layers = nn.ModuleList(
        #     [BertCrossLayer(bert_config) for _ in range(config['num_end_layer'])])
        # self.cross_modal_smiles_layers.apply(utils.init_weights)

        # header predict layer
        # 这段代码配置了头部预测层，包括跨模态图像和 SMILES 的池化层，并设置了掩码语言模型 (Masked Language Model, MLM) 的概率和掩码比例。
        # cross_modal_image_pooler 和 cross_modal_smiles_pooler 是用于对图像和 SMILES 特征进行池化的层，将其转换为固定长度的向量表示。这有助于将变长的输入序列转换为固定长度的表示形式。
        self.cross_modal_image_pooler = predictor.Pooler(config["hidden_size"])
        self.cross_modal_image_pooler.apply(utils.init_weights)
        self.cross_modal_smiles_pooler = predictor.Pooler(config["hidden_size"])
        self.cross_modal_smiles_pooler.apply(utils.init_weights)
        self.cross_modal_graph_pooler = predictor.Pooler(config["hidden_size"])
        self.cross_modal_graph_pooler.apply(utils.init_weights)
        self.cross_modal_pos_pooler = predictor.Pooler(config["hidden_size"])
        self.cross_modal_pos_pooler.apply(utils.init_weights)

        # self.cross_modal_image_pooler1 = predictor.Pooler1(config["hidden_size"])
        # self.cross_modal_image_pooler1.apply(utils.init_weights)
        # self.cross_modal_smiles_pooler1 = predictor.Pooler1(config["hidden_size"])
        # self.cross_modal_smiles_pooler1.apply(utils.init_weights)
        # self.cross_modal_graph_pooler1 = predictor.Pooler1(config["hidden_size"])
        # self.cross_modal_graph_pooler1.apply(utils.init_weights)

        # self.gate_weights = nn.Linear(cls_feats_all.size(-1), cls_feats_all.size(-1))
        # self.gate_weights = nn.Linear(config["hidden_size"]*4, config["hidden_size"]*4)#add 门控机制来动态加权不同模态的贡献

        # mlm_probability 是掩码语言模型的概率，用于控制在输入的 SMILES 序列中生成掩码的频率。掩码语言模型是一种预训练模型，用于预测序列中被遮盖的标记。
        self.mlm_probability = config['mlm_probability']
        self.mask_ratio = config['mask_ratio']  # mask_ratio 是掩码比例，用于控制输入 SMILES 序列中被掩码的比例。掩码比例表示在输入序列中应该被掩码的标记比例。

        # 这段代码根据配置中的损失名称和权重，初始化了不同的头部预测层，用于不同的损失任务。
        # MLM损失 #如果配置中包含 MLM 损失，则初始化 MLM 头部预测层 mlm_score。
        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = predictor.MLMHead(bert_config_encoder)
            self.mlm_score.apply(utils.init_weights)
        # itm损失 or ocsr下游任务#如果配置中包含 itm（图像-文本匹配）损失、ocsr（图像分类）损失或 ocsr_finturn（微调的图像分类）损失，则初始化 itm 头部预测层 itm_score。
        if config["loss_names"]["itm"] > 0 or config["loss_names"]["ocsr"] > 0 or config["loss_names"][
            "ocsr_finturn"] > 0:
            self.itm_score = predictor.ITMHead(config["hidden_size"] * 2)
            # self.itm_score = predictor.ITMHead(12)
            self.itm_score.apply(utils.init_weights)

        # fpp损失#如果配置中包含 fpp（指纹预测）损失，则初始化 fpp 头部预测层。针对不同规模的指纹（100、1000、10000），分别初始化相应的头部预测层。
        if config["loss_names"]["fpp"] > 0:
            self.fpp_score_smiles100 = predictor.FPPHead(config["hidden_size"], 100)
            self.fpp_score_smiles100.apply(utils.init_weights)
            self.fpp_score_images100 = predictor.FPPHead(config["hidden_size"], 100)
            self.fpp_score_images100.apply(utils.init_weights)

            self.fpp_score_smiles1000 = predictor.FPPHead(config["hidden_size"], 1000)
            self.fpp_score_smiles1000.apply(utils.init_weights)
            self.fpp_score_images1000 = predictor.FPPHead(config["hidden_size"], 1000)
            self.fpp_score_images1000.apply(utils.init_weights)

            self.fpp_score_smiles10000 = predictor.FPPHead(config["hidden_size"], 10000)
            self.fpp_score_smiles10000.apply(utils.init_weights)
            self.fpp_score_images10000 = predictor.FPPHead(config["hidden_size"], 10000)

            self.fpp_score_images10000.apply(utils.init_weights)

        # ===================== Downstream task ===================== #

        if (
                self.hparams.config["load_path"] != ""
                and not self.hparams.config["test_only"] and not self.hparams.config['is_pretrain']
        ):
            print('===================== load checkpoint for downstream task =====================')
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

        if config["loss_names"]["MoleculeNet_classify"] > 0:
            self.MoleculeNet_classify_score = predictor.MoleculeClassify(config["hidden_size"] * 4,
                                                                         drop_rate=config['drop_rate'])
            # self.MoleculeNet_classify_score = predictor.MoleculeClassify(16,drop_rate=config['drop_rate'])
            self.MoleculeNet_classify_score.apply(utils.init_weights)

        if config['loss_names']['MoleculeNet_regress'] > 0:
            self.MoleculeNet_regress_score = predictor.MoleculeRegress(config['hidden_size'] * 4
                                                                       ,drop_rate=config['drop_rate'])
            self.MoleculeNet_regress_score.apply(utils.init_weights)

        self.current_tasks = list()
        # train/loss
        metrics_utils.set_metrics(self)
        metrics_utils.set_task(self)

        self.focal_loss = FocalLoss(ignore_index=-100)
        self.criterion = ContrastiveLoss(loss_computer='nce_softmax', temperature=args.temperature, args=args).cuda()
        # ===================== test_only ======================
        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            print('===================== load checkpoint for test only =====================')
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            msg = self.load_state_dict(state_dict, strict=False)
            # print(msg)
        if self.hparams.config["load_path"] == "" and ~self.hparams.config["test_only"]:
            print('===================== fineturning with nonPretrained =====================')

    def _init_weight(self):
        nn.init.normal_(self.global_init, mean=0, std=self.latent_size ** -0.5)

    # 这个 infer 函数主要用于推断阶段的处理，逐步解释其中的步骤：
    # 参数解析：函数接收多个参数，包括图像 images、SMILES 字符串 smiles，以及一些可选参数，如负样本图像 images_false、负样本 SMILES 字符串 smiles_false、图像的 token 类型索引 image_token_type_idx、是否对图像进行掩码 mask_image、是否对 SMILES 进行掩码 mask_smiles 以及是否改变 SMILES 输入 change_smi。
    def infer(
            self, images, smiles, images_false=None, smiles_false=None,
            image_token_type_idx=1,
            graph_token_type_idx=2,
            pos_token_type_idx=3,
            mask_image=False,
            mask_smiles=False,
            change_smi=False,  # add
            data=False,
            mode="mask"
    ):

        # device = images[0].device
        device = images.device
        # 转换数据格式：将输入的 SMILES 字符串转换为列表形式，方便后续处理。如果提供了负样本 SMILES 列表，则也将其转换为列表格式。

        '''
        #add3d
        #print(f'x',len(data.x))
        # 将 data 中的所有相关数据转换为 Tensor
        x = torch.cat([torch.tensor(xi, dtype=torch.int64) for xi in data.x], dim=0).to(device)  # 节点特征
        edge_index = torch.cat([torch.tensor(ei, dtype=torch.int64) for ei in data.edge_index], dim=1).to(device)  # 边索引
        edge_attr = torch.cat([torch.tensor(ei, dtype=torch.int64) for ei in data.edge_attr], dim=0).to(device)  # 边属性

        #node_batch = [torch.tensor(ei, dtype=torch.long) for ei in data.batch]

        node_batch = torch.cat([torch.tensor(ei, dtype=torch.long) for ei in data.batch], dim=0).to(device)  # 节点批次
        face_mask = torch.cat([torch.tensor(ei, dtype=torch.bool) for ei in data.ring_mask], dim=0).to(device)  # 面掩码
        face_index = torch.cat([torch.tensor(ei, dtype=torch.int64) for ei in data.ring_index], dim=1).to(device)  # 面索引
        print(f"data.nf_node",data.nf_node)
        #nf_node = torch.cat([torch.tensor(ei, dtype=torch.int64) for ei in data.nf_node], dim=0).to(device)  # 节点的特征
        #nf_face = torch.cat([torch.tensor(ei, dtype=torch.int64) for ei in data.nf_ring], dim=0).to(device)  # 面的特征
        pos = torch.cat([torch.tensor(ei, dtype=torch.float) for ei in data.pos], dim=0).to(device)  # 3D坐标

        # 将标量转换为 Tensor
        num_nodes = torch.tensor(data.n_nodes, dtype=torch.long).to(device)
        num_faces = torch.tensor(data.num_rings, dtype=torch.long).to(device)
        num_edges = torch.tensor(data.n_edges, dtype=torch.long).to(device)
        num_graphs = torch.tensor(data.num_graphs, dtype=torch.long).to(device)


        #print(f"data",data)
        '''
        x = data.x.to(device)

        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_attr.to(device)
        node_batch = data.batch.to(device)
        face_mask = data.ring_mask.to(device)
        face_index = data.ring_index.to(device)

        pos = data.pos.to(device)
        # print(f"pos",pos.shape)#pos torch.Size([264, 3])
        num_nodes = data.n_nodes.to(device)
        num_faces = data.num_rings.to(device)
        num_edges = data.n_edges.to(device)
        num_graphs = data.num_graphs
        # isomorphisms = data.isomorphisms
        # print(f'isomorphisms',isomorphisms)
        # nf_node = data.nf_node
        # nf_face = data.nf_ring
        nf_node = torch.cat([torch.tensor(ei, dtype=torch.int64) for ei in data.nf_node], dim=1).to(device)  # 节点的特征
        nf_face = torch.cat([torch.tensor(ei, dtype=torch.int64) for ei in data.nf_ring], dim=1).to(device)
        #        print(f'x:',x.shape)
        #        print(f'pos:',pos.shape)
        #        print(f'node_batch:',node_batch.shape)

        pos_mean = global_mean_pool(pos, node_batch)

        pos = pos - torch.repeat_interleave(pos_mean, num_nodes, dim=0)
        if args.random_rotation:
            pos = get_random_rotation_3d(pos)
        x, attr_mask_index = self.node_embedding(x, mode=mode)
        edge_attr = one_hot_bonds(edge_attr)
        edge_attr = self.encoder_edge(edge_attr)
        edge_attr = self.node_embedding.update_edge_feat(
            edge_attr, edge_index, attr_mask_index, mode=mode
        )

        graph_idx = torch.arange(num_graphs).to(x.device)
        edge_batch = torch.repeat_interleave(graph_idx, num_edges, dim=0)
        u = self.global_init.expand(num_graphs, -1)

        if self.use_face:
            face_batch = torch.repeat_interleave(graph_idx, num_faces, dim=0)
            node_attributes = self.aggregate_edges_for_face_fn(
                x[nf_node], nf_face, size=num_faces.sum().item()
            )
            sent_attributes = self.aggregate_edges_for_face_fn(
                edge_attr, face_index[0], size=num_faces.sum().item()
            )
            received_attributes = self.aggregate_edges_for_face_fn(
                edge_attr, face_index[1], size=num_faces.sum().item()
            )
            feat = torch.cat([node_attributes, sent_attributes, received_attributes], dim=1)
            feat = torch.where(face_mask.unsqueeze(1), feat.new_zeros((feat.shape[0], 1)), feat)
            face = self.encoder_face(feat)
        else:
            face = None
            face_batch = None
            face_index = None

        # print(f'x',x.shape)#torch.Size([224, 256])
        # print(f'pos', pos.shape)#torch.Size([224, 3])
        x, edge_attr, face, u, pos_predictions, pos_mask_idx = self.gnn_layers(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            u=u,
            node_batch=node_batch,
            edge_batch=edge_batch,
            num_nodes=num_nodes,
            num_edges=num_edges,
            face=face,
            face_batch=face_batch,
            face_mask=face_mask,
            face_index=face_index,
            num_faces=num_faces,
            nf_node=nf_node,
            nf_face=nf_face,
            mode=mode,
            pos=pos,
        )

        pos_embed = self.pooling(x, node_batch, size=num_graphs)  # gnn_x torch.Size([12, 768])
        pos_embed = GradMultiply.apply(pos_embed, self.gradmultiply)
        pos_embed = self.attr_decoder(pos_embed, mode=mode)
        if mode == "raw":  # self.attr_predict:torch.Size([224, 768])-->torch.Size([12, 768])
            x = self.pooling(x, node_batch, size=num_graphs)  # gnn_x torch.Size([12, 768])
            x = GradMultiply.apply(x, self.gradmultiply)
        # print(f'gnn_x', x.shape)#gnn_x torch.Size([224, 768])
        pred_attrs = self.attr_decoder(x, mode=mode)

        # pos_predictions = torch.stack(pos_predictions, dim=0)
        # print(f"pred_attrs",pred_attrs[0].shape)#pred_attrs torch.Size([264, 768])
        # print(f"attr_mask_index", attr_mask_index.shape)#attr_mask_index torch.Size([264, 1])
        # print(f"pos_predictions", pos_predictions[0].shape)#pos_predictions torch.Size([264, 3])
        # print(f"pos_mask_idx", pos_mask_idx.shape)#pos_mask_idx torch.Size([264, 1])
        '''
            if mode == "mol2conf":
                return None, None, pos_predictions, None
            pred_attrs = self.attr_decoder(x, mode=mode)
            if mode == "conf2mol":
                return pred_attrs, None, None, None
            return pred_attrs, attr_mask_index, pos_predictions, pos_mask_idx
            '''

        # finish 3d

        smiles = list(smiles)  # tuple->list
        # smiles1 = ''.join([char.upper() if char.islower() else char for char in smiles])
        # smiles1=list(smiles1)
        # print(smiles1)
        # print(smiles)
        # print(smiles)

        if smiles_false != None:
            smiles_false = list(smiles_false)
        # data = get_data(path=args.data_path, args=args, logger=logger)#获取 SMILES 字符串和目标数值，并将它们封装成一个 MoleculeDataset 对象。
        # data = get_data_from_smiles(smiles_list=smiles, logger=logger)  # add
        data = get_data(smiles_list=smiles, args=args)
        '''
                if args.dataset_type == 'regression':
                    train_smiles, train_targets = data.smiles(), data.targets()
                    scaler = StandardScaler().fit(train_targets)
                    scaled_targets = scaler.transform(train_targets).tolist()
                    train_data.set_targets(scaled_targets)

                else:
                    scaler = None
                '''
        # MoleculeDataset(data[i:i + args.batch_size])
        mol = MoleculeDataset(data)
        smiles1, features = mol.smiles(), mol.features()
        # print(smiles1)
        if args.step == 'pretrain':
            step = 'pretrain'
        else:
            step = 'finetune'
        # step  = 'functional_prompt'
        '''
        graph_model = self.graph_model()
        graph_model.encoder.load_state_dict(torch.load(/home/unionlab001/CXX/my/KANO/dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl, map_location='cpu'), strict=False)
        add_functional_prompt(graph_model, args)
        graph_emb = graph_model(step, True, smiles, None)
        print(smiles1)
        print(smiles)
        '''

        graph_embedding1 = self.graph_encoder(step, False, smiles1, None)  # num_attention_heads=config[
        graph_embedding2 = self.graph_encoder2(step, True, smiles1, None)

        # print(images.shape)torch.Size([12, 3, 224, 224])

        # print(graph_embedding1.shape)#torch.Size([12, 3, 300])
        graph_embedding = self.cross_modal_graph_transform(graph_embedding1)
        # graph_embedding = self.cross_modal_graph_transform(graph_embedding2)

        graph_masks = torch.ones((graph_embedding.size(0), graph_embedding.size(1)), dtype=torch.long, device=device)
        extend_graph_masks = self.smiles_encoder.get_extended_attention_mask(graph_masks, graph_masks.size(), device)

        if change_smi == True:  # 处理改变 SMILES 输入：如果 change_smi 参数为 True，则表示要改变输入的 SMILES。在这种情况下，函数会执行以下操作：
            # input_ids = smiles.input_ids.clone()
            # input_ids, s_labels = self.change_smi(input_ids, self.smiles_encoder.config.vocab_size, images.device)
            # s_labels = s_labels.to(images.device)
            # smiles.input_ids = input_ids
            #
            # pos_len = images.size(0) // 2
            # neg_len = images.size(0) - pos_len
            # i_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(images.device)
            # # 随机打乱
            # i_labels = i_labels[torch.randperm(i_labels.size(0))] # 随机打乱
            #
            # s = i_labels[:,None,None,None]
            # images = images * s + images_false * (1 - s)
            # # ocsr_labels = i_labels
            #
            # ocsr_labels = s_labels * i_labels # 合并

            # input_ids = smiles.input_ids.clone()
            # input_ids,images,ocsr_labels = self.shufflePair(input_ids,images,images_false,self.smiles_encoder.config.vocab_size,device)
            # smiles.input_ids = input_ids
            labels = torch.randint(0, 3, (images.shape[0],))  # 随机生成标签，用于标识哪些样本需要更改 SMILES 或图像。
            # 如果标签为 1，则将图像替换为负样本图像。
            index_1 = torch.nonzero(labels == 1).squeeze(1)
            if index_1.shape[0] != 0:
                images[index_1] = images_false[index_1]

            index_2 = torch.nonzero(labels == 2).squeeze(1)
            # 如果标签为 2，则将 SMILES 替换为负样本 SMILES。
            if index_2.shape[0] != 0:
                for i in index_2:
                    smiles[i] = smiles_false[i]
            ocsr_labels = 1 - labels.bool().float()
        else:
            ocsr_labels = None
        lengths = [len(cap) for cap in smiles]
        # print(lengths)

        # 处理 SMILES 数据：使用预训练的 tokenizer 将 SMILES 字符串转换为模型可接受的格式。
        smiles = self.tokenizer(smiles, padding='max_length', truncation=True, max_length=202,
                                return_tensors="pt").to(device)
        # 如果 mask_smiles 参数为 True，则表示要对 SMILES 进行掩码处理，函数会根据指定的概率对输入的 SMILES 进行掩码操作，并返回掩码后的输入以及对应的掩码标签。否则，不执行掩码处理。
        if mask_smiles == True:
            input_ids = smiles.input_ids.clone()
            mlm_labels = input_ids.clone()
            probability_matrix = torch.full(mlm_labels.shape, self.mlm_probability)
            input_ids, mlm_labels = self.mask(input_ids, self.smiles_encoder.config.vocab_size, images.device,
                                              targets=mlm_labels,
                                              probability_matrix=probability_matrix)
            smiles.input_ids = input_ids
        else:
            mlm_labels = None

        # .logits (bs,max_length,input_smiles_embed_size)
        smiles_embedding = self.smiles_encoder(smiles.input_ids, attention_mask=smiles.attention_mask, return_dict=True)
        # (bs,max_length,hidden_size)
        # print(smiles_embedding.logits)#torch.Size([12, 202, 128])
        smiles_embedding = self.cross_modal_smiles_transform(smiles_embedding.logits)
        # (bs,max_length) 全为1
        smiles_masks = torch.ones((smiles_embedding.size(0), smiles_embedding.size(1)), dtype=torch.long, device=device)
        # (bs,1,1,max_len) 全为0
        extend_smiles_masks = self.smiles_encoder.get_extended_attention_mask(smiles_masks, smiles_masks.size(), device)
        # (bs,patchsize,hidden_size)

        # 处理图像数据：通过调用视觉编码器（Visual Encoder）对图像进行编码，并根据参数确定是否对图像进行掩码处理。如果掩码图像，函数会返回掩码后的图像数据、掩码、以及用于恢复原始形状的索引。否则，不执行掩码处理。
        if mask_image == False:
            # mask : None
            # ids_restore : None
            image_embedding, mask, ids_restore = self.visual_encoder.forward_encoder(images, mask_ratio=0)
        else:
            image_embedding, mask, ids_restore = self.visual_encoder.forward_encoder(images, mask_ratio=self.mask_ratio)
        # print(image_embedding)#torch.Size([12, 197, 768])
        # 4,167,768
        image_masks = torch.ones((image_embedding.size(0), image_embedding.size(1)), dtype=torch.long,
                                 device=device)
        image_embedding = self.cross_modal_image_transform(image_embedding)

        # (4,1,1,197)
        extend_image_masks = self.smiles_encoder.get_extended_attention_mask(image_masks, image_masks.size(),
                                                                             device=device)
        # cross-model
        # image_embeds(4,30,768)

        # print(smiles_embedding.shape)
        # print(graph_embedding.shape)
        # print(image_embedding.shape)
        # graph_embeds, image_embeds = (
        #     graph_embedding + self.token_type_embeddings(torch.zeros_like(graph_masks)),
        #     image_embedding
        #     + self.token_type_embeddings(
        #         torch.full_like(image_masks.long(), image_token_type_idx)
        #     ),
        # )
        #
        # # (4,3,768) (4,197,768)
        # y, z = graph_embeds, image_embeds
        # for graph_layer, image_layer in zip(self.cross_modal_graph_layers, self.cross_modal_image_layers):
        #     y1 = graph_layer(y, z, extend_graph_masks, extend_image_masks)
        #     z1 = image_layer(z, y, extend_image_masks, extend_graph_masks)
        #     y, z = y1[0], z1[0]
        # # (bs,num_graph_nodes,hidden_size) (bs,patch_nums+1,hidden_size)
        # # graph_feats, image_feats = y, z
        #
        # # normalize,pair特征
        # graph_feats = F.normalize(y, dim=-1)
        # image_feats = F.normalize(z, dim=-1)
        # # print(graph_feats.shape, image_feats.shape)

        # added
        smiles_embeds, graph_embeds = (
            smiles_embedding + self.token_type_embeddings(torch.zeros_like(smiles_masks)),
            graph_embedding
            + self.token_type_embeddings(
                torch.full_like(graph_masks.long(), image_token_type_idx)  # gai
            ),
        )

        # (4,30,768) (4,3,768)
        x, z = smiles_embeds, graph_embeds
        for smiles_layer, graph_layer in zip(self.cross_modal_smiles_layers, self.cross_modal_graph_layers):
            x1 = smiles_layer(x, z, extend_smiles_masks, extend_graph_masks)
            z1 = graph_layer(z, x, extend_graph_masks, extend_smiles_masks)
            x, z = x1[0], z1[0]

        # (bs,max_len,hidden_size) (bs,num_graph_nodes,hidden_size)
        # smiles_feats, graph_feats = x, z

        # normalize,pair特征
        smiles_feats = F.normalize(x, dim=-1)  # smiles_feats
        graph_feats = F.normalize(z, dim=-1)
        # print(smiles_feats.shape, graph_feats.shape)

        # 交叉模态编码：通过 SMILES 编码器和图像编码器，将 SMILES 和图像数据进行交叉模态编码，得到归一化后的特征表示。
        smiles_embeds, image_embeds = (
            smiles_embedding + self.token_type_embeddings(torch.zeros_like(smiles_masks)),
            image_embedding
            + self.token_type_embeddings(
                torch.full_like(image_masks.long(), image_token_type_idx)
            ),
        )

        # (4,30,768) (4,197,768)
        x, y = smiles_embeds, image_embeds
        for smiles_layer, image_layer in zip(self.cross_modal_smiles_layers, self.cross_modal_image_layers):
            x1 = smiles_layer(x, y, extend_smiles_masks, extend_image_masks)
            y1 = image_layer(y, x, extend_image_masks, extend_smiles_masks)
            x, y = x1[0], y1[0]
        # (bs,max_len,hidden_size) (bs,patch_nums+1,hidden_size)
        # smiles_feats, image_feats = x, y

        # normalize,pair特征
        # smiles_embedding = F.normalize(x, dim=-1)
        smiles_feats = F.normalize(x, dim=-1)
        image_feats = F.normalize(y, dim=-1)
        # print(smiles_feats.shape, image_feats.shape)

        if mask_image:
            pred = self.visual_encoder.forward_decoder(image_feats, ids_restore)
            patchify_img = self.visual_encoder.patchify(images)
        else:
            pred = None
            patchify_img = None

        # 池化层
        # 这段代码负责对特征进行池化操作，生成用于分类或回归的最终特征表示，并将各种处理后的数据和结果返回。
        cls_feats_smiles = self.cross_modal_smiles_pooler(smiles_feats)
        cls_feats_image = self.cross_modal_image_pooler(
            image_feats)  # 池化操作：通过交叉模态池化层，分别对 SMILES 特征和图像特征进行池化操作，得到两个池化后的特征表示 cls_feats_smiles 和 cls_feats_image。
        cls_feats_graph = self.cross_modal_graph_pooler(graph_feats)

        # ot_image = compute_ot(cls_feats_smiles, cls_feats_image)  # Image 和 SMILES 的 OT
        # ot_graph = compute_ot(cls_feats_smiles, cls_feats_graph)  # Graph 和 SMILES 的 OT
        # ot_pos = compute_ot(cls_feats_smiles, cls_feats_pos)      # Pos 和 SMILES 的 OT

        # Step 3: 使用 OT 矩阵对齐每个模态的特征，并反向对齐 cls_feats_smiles
        # cls_feats_image = torch.matmul(ot_image, cls_feats_image)  # 对齐 Image 特征
        # cls_feats_graph = torch.matmul(ot_graph, cls_feats_graph)  # 对齐 Graph 特征
        # cls_feats_pos = torch.matmul(ot_pos, cls_feats_pos)        # 对齐 Pos 特征

        # 双向对齐，使用 OT 矩阵的转置来对齐 SMILES 特征
        # cls_feats_smiles = torch.matmul(ot_image.transpose(0, 1), cls_feats_smiles)  # 对齐 SMILES 特征到 Image

        cls_feats = torch.cat([cls_feats_image, cls_feats_smiles],
                              dim=-1)  # 合并特征：将池化后的 SMILES 特征和图像特征按照最后一个维度（特征维度）拼接起来，得到最终的特征表示 cls_feats。

        # cls_feats_graph = graph_feats
        # cls_feats_all = torch.cat([cls_feats_image, cls_feats_smiles, cls_feats_graph], dim=-1)

        if True:  # mode == "raw":
            # pos_embedding = torch.stack(pred_attrs, dim=1)
            pos_embedding = torch.stack(pos_embed, dim=1)

            # print(pos_embedding.shape)

            pos_masks = torch.ones((pos_embedding.size(0), pos_embedding.size(1)), dtype=torch.long, device=device)

            extend_pos_masks = self.smiles_encoder.get_extended_attention_mask(pos_masks, pos_masks.size(), device)

            smiles_embeds, pos_embeds = (
                smiles_embedding + self.token_type_embeddings(torch.zeros_like(smiles_masks)),
                pos_embedding
                + self.token_type_embeddings(
                    torch.full_like(pos_masks.long(), image_token_type_idx)
                ),
            )

            # (4,30,768) (4,197,768)
            x, w = smiles_embeds, pos_embeds
            for smiles_layer, pos_layer in zip(self.cross_modal_smiles_layers, self.cross_modal_pos_layers):
                x1 = smiles_layer(x, w, extend_smiles_masks, extend_pos_masks)
                w1 = pos_layer(w, x, extend_pos_masks, extend_smiles_masks)
                x, w = x1[0], w1[0]
            # (bs,max_len,hidden_size) (bs,patch_nums+1,hidden_size)
            # smiles_feats, image_feats = x, y

            # normalize,pair特征
            smiles_feats = F.normalize(x, dim=-1)
            pos_feats = F.normalize(w, dim=-1)
            # print(smiles_feats.shape, image_feats.shape)
            cls_feats_pos = self.cross_modal_pos_pooler(pos_feats)
            # cls_feats_smiles = self.cross_modal_smiles_pooler(smiles_feats)

            # ot_pos = compute_ot(cls_feats_smiles, cls_feats_pos)      # Pos 和 SMILES 的 OT
            # cls_feats_pos = torch.matmul(ot_pos, cls_feats_pos)        # 对齐 Pos 特征

            # cls_feats_pos = self.cross_modal_pos_pooler(pos_embedding)
            #print(f'cls_feats_pos', cls_feats_pos.shape)

            cls_feats_all = torch.cat([cls_feats_image, cls_feats_smiles, cls_feats_graph, cls_feats_pos], dim=-1)

            # cls_feats_all = cls_feats_all * torch.sigmoid(self.gate_weights(cls_feats_all))






        else:
            cls_feats_all = torch.cat([cls_feats_image, cls_feats_smiles, cls_feats_graph], dim=-1)
        # print(f"smiles_feats: {smiles_feats.grad_fn}")
        # print(f"image_feats: {image_feats.grad_fn}")
        # print(f"cls_feats_graph: {cls_feats_graph.grad_fn}")
        # print(f"cls_feats_pos: {cls_feats_pos.grad_fn}")


        ret = {
            "smiles_feats": smiles_feats,  # torch.Size([12, 202, 768])
            "image_feats": image_feats,  # torch.Size([12, 197, 768])
            "graph_feats": graph_feats,
            "cls_feats": cls_feats,  # torch.Size([12, 1536])
            "cls_feats_smiles": cls_feats_smiles,
            "cls_feats_image": cls_feats_image,
            # "cls_feats_smiles1": cls_feats_smiles1,  # torch.Size([12, 202, 768])
            # "cls_feats_image1": cls_feats_image1,  # torch.Size([12, 197, 768])
            "cls_feats_graph": cls_feats_graph,
            "cls_feats_pos": cls_feats_pos,
            "cls_feats_all": cls_feats_all,
            'smiles_masks': smiles.attention_mask,
            "image_masks": image_masks,
            "ids_restore": ids_restore,
            "mask": mask,
            "pred": pred,
            "imgs": patchify_img,
            "graph_emb1": graph_embedding1,
            "graph_emb2": graph_embedding2,
            "mlm_labels": mlm_labels,
            "ocsr_labels": ocsr_labels,
            "lengths": lengths,
            "pred_attrs": pred_attrs,
            "attr_mask_index": attr_mask_index,
            "pos_predictions": pos_predictions,
            "pos_mask_idx": pos_mask_idx,

        }
        return ret  # 结果返回：将处理后的各种数据和结果以字典形式返回，其中包括原始的 SMILES 和图像特征、池化后的特征、注意力掩码、恢复的图像补丁、掩码、预测结果等。

    def forward(self, batch, testing=False):
        # smiles = {input_ids,token_type_ids,attention_mask}
        print(f'self.current_tasks', self.current_tasks)

        if 'MoleculeNet_classify' in self.current_tasks:
            smiles, images, label, pos = batch
        elif 'MoleculeNet_regress' in self.current_tasks:
            smiles, images, label, pos = batch


        else:
            images, _, smiles, _, _, _, _, pos = batch
        mol = list(smiles)
        pos = list(pos)

        data = process_data(mol, pos, smiles2graphwithface, isomorphic_core)
        # print(f'data:{data.num_graphs}')
        '''
        images, _, smiles, *el = batch
        mol = list(smiles)


        data = process_data(mol,  smiles2graphwithface, isomorphic_core)
        # print(f'data:{data.num_graphs}')
        '''
        ret = dict()
        #self.current_tasks=["MoleculeNet_regress"]
        print(f'self.current_tasks', self.current_tasks)

        if len(self.current_tasks) == 0:
            images, smiles = batch
            ret.update(self.infer(images, smiles))
            return ret
        ####============================== Pretrain task ==================================####
        ## =========================== Graph contrastive learning =========================== ##
        if 'gcl' in self.current_tasks:
            # start_time = time.time()  # Start timer
            ret.update(logger_gcl(self, batch, data))
            # print(f"Task 'gcl' took {time.time() - start_time:.2f} seconds.")

        ## =========================== Image-Smiles Matching =========================== ##
        # Image-Smiles Matching
        if "itm" in self.current_tasks:
            # start_time = time.time()  # Start timer

            ret.update(logger_itm(self, batch, data))
            # print(f"Task 'itm' took {time.time() - start_time:.2f} seconds.")

        ## =========================== Mask Language Modeling =========================== ##
        if "mlm" in self.current_tasks:
            # start_time = time.time()  # Start timer
            ret.update(logger_mlm(self, batch, data))
            # print(f"Task 'mlm' took {time.time() - start_time:.2f} seconds.")

        ## =========================== Mask Patches Modeling =========================== ##
        if "mpp" in self.current_tasks:
            # start_time = time.time()  # Start timer
            ret.update(logger_mpp(self, batch, data))
            # print(f"Task 'mpp' took {time.time() - start_time:.2f} seconds.")

        ## =========================== fingerprint predict =========================== ##
        if 'fpp' in self.current_tasks:
            # start_time = time.time()  # Start timer
            ret.update(logger_fpp(self, batch, data))
            # print(f"Task 'fpp' took {time.time() - start_time:.2f} seconds.")
        ## ===================== joint_embedding_model ======================= ##
        if 'jem' in self.current_tasks:
            # start_time = time.time()  # Start timer
            ret.update(joint_embedding_model(self, batch, data))
            # print(f"Task 'jem' took {time.time() - start_time:.2f} seconds.")

        ## ===================== mask 2dand3d_model ======================= ##
        # if 'm2d' in self.current_tasks:
        #     # start_time = time.time()  # Start timer
        #     ret.update(mask2d(self, batch, data))
        #     # print(f"Task 'm2d' took {time.time() - start_time:.2f} seconds.")
        #
        #     ## ===================== conf2mol_model ======================= ##
        # if 'c2m' in self.current_tasks:
        #     # start_time = time.time()  # Start timer
        #     ret.update(conf2mol(self, batch, data))
        #     # print(f"Task 'c2m' took {time.time() - start_time:.2f} seconds.")
        #
        # ## ===================== mol2conf_model ======================= ##
        # if 'm2c' in self.current_tasks:
        #     # start_time = time.time()  # Start timer
        #     ret.update(mol2conf(self, batch, data))
        #     # print(f"Task 'm2c' took {time.time() - start_time:.2f} seconds.")

        ##=========================== graph predict =========================== ##

        ####============================== downstream task ==================================####

        ## =========================== MoleculeNet classification =========================== ##
        if 'MoleculeNet_classify' in self.current_tasks:
            ret.update(compute_MoleculeNet_classify(self, batch, data, testing))

        if 'MoleculeNet_regress' in self.current_tasks:
            ret.update(compute_MoleculeNet_regress(self, batch, data, testing))

        if 'ocsr_finturn' in self.current_tasks:
            ret.update(compute_ocsr_finturn(self, batch))

        if 'ocsr' in self.current_tasks:
            ret.update(compute_ocsr(self, batch))

        if 'inter' in self.current_tasks:
            ret.update(Uamp(self, batch))

        if 'auroc' in self.current_tasks:
            ret.update(write_auroc(self, batch))

        return ret

    def training_step(self, batch, batch_idx):  # training_step: 定义了每个训练步骤的操作。调用模型的前向传播方法来计算输出，然后根据输出中包含的损失值计算总损失并返回。
        output = self(batch)

        total_loss = sum([v for k, v in output.items() if "loss" in k])#.requires_grad_(True)

        print(total_loss.requires_grad, total_loss.grad_fn)  # 应该是 True 和 非 None
        return total_loss

    def training_epoch_end(self, outs):  # training_epoch_end: 定义了训练周期结束时的操作。在此处执行了一些关于训练指标的收尾工作。
        metrics_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):  # validation_step: 定义了每个验证步骤的操作。与训练步骤类似，但在返回总损失之前不执行损失的加和操作。
        metrics_utils.set_task(self)
        output = self(batch)
        return sum([v for k, v in output.items() if "loss" in k])

    def validation_epoch_end(self, outs):  # validation_epoch_end: 定义了验证周期结束时的操作。在此处执行了与训练周期结束时类似的验证指标的收尾工作。
        metrics_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):  # test_step: 定义了每个测试步骤的操作。与训练和验证步骤类似，但如果模型不是用于预训练，则还会计算 AUROC 指标。
        metrics_utils.set_task(self)
        output = self(batch, testing=True)
        total_loss = sum([v for k, v in output.items() if "loss" in k])
        return total_loss

    def test_epoch_end(self, outs):  # test_epoch_end: 定义了测试周期结束时的操作。如果模型不是用于预训练，则执行测试周期的 AUROC 计算；否则，执行与训练周期结束时类似的收尾工作。
        if self.config['is_pretrain'] == False:
            metrics_utils.test_epoch_auroc(self)
        else:  # 预训练
            metrics_utils.epoch_wrapup(self)

    # 这段代码包含了用于模型预测和优化器配置的方法，以及用于初始化 tokenizer 的辅助方法。
    def predict_step(self, batch, batch_idx,
                     dataloader_idx=0):  # predict_step: 定义了每个预测步骤的操作。在这里，模型直接调用前向传播方法对输入进行预测并返回输出。
        return self(batch)

    def configure_optimizers(
            self):  # configure_optimizers: 定义了优化器的配置方法。在这里，调用了一个辅助函数 utils.set_schedule(self) 来设置优化器的学习率调度器，并返回配置好的优化器。
        return utils.set_schedule(self)

    def get_pretrained_tokenizer(self,
                                 tokenizer):  # get_pretrained_tokenizer: 辅助方法，用于获取预训练的 tokenizer。如果是分布式训练且当前进程是主进程，则在主进程中获取 tokenizer，并在所有进程间进行同步；否则，直接在当前进程获取 tokenizer。
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                # 主线程
                AutoTokenizer.from_pretrained(tokenizer)
                # AutoTokenizer.from_pretrained("./DeepChem/ChemBERTa-77M-MLM")
            torch.distributed.barrier()

        return AutoTokenizer.from_pretrained(tokenizer)
        # return AutoTokenizer.from_pretrained("./DeepChem/ChemBERTa-77M-MLM")

    def init_tokenizer(self):  # s初始化 tokenizer 的方法。在这里调用了 get_pretrained_tokenizer 方法来获取预训练的 tokenizer。
        tokenizer = self.get_pretrained_tokenizer(self.config['tokenizer'])
        return tokenizer

    # 预训练模型通常使用特定的 tokenizer，例如 BERT 模型使用的是 WordPiece tokenizer，GPT 模型使用的是 Byte-Pair Encoding (BPE) tokenizer。这些 tokenizer 能够将文本分割成单词或子词，并将其转换为模型可以理解的格式，例如输入 token IDs、attention masks 等。
    # 在深度学习中，tokenizer 是将原始文本数据转换为模型输入数据的重要预处理步骤之一。

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None,
             probability_matrix=None):  # 这个 mask 方法用于对输入的文本数据进行遮盖操作，以模拟 Masked Language Modeling (MLM) 的预训练任务。
        if masked_indices is None:
            # probability_matrix (bs,max_len) 0.15
            # (bs,max_len) bool
            masked_indices = torch.bernoulli(
                probability_matrix).bool()  # 根据给定的概率矩阵 probability_matrix，随机生成要遮盖的 token 的索引。
        # pad_token_id=0,cls_token_id=12,mask_token_id=14
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[
            input_ids == self.tokenizer.cls_token_id] = False  # 将特殊 token（如 padding token 和 cls token）的索引从遮盖的范围中排除。
        if targets is not None:  # 如果提供了 targets 参数（即要预测的 token），则将未遮盖的 token 对应的 targets 设为 -100（这些 token 不参与损失的计算）。
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 将 80% 的遮盖的 token 替换为特殊的 mask token（通常为 [MASK]）。
        # 将剩余的 10% 的遮盖的 token 替换为随机选择的词。
        # 剩下的 10% 的 token 保持不变。
        # 最终，该方法会返回处理后的输入 token IDs，以及可能的预测目标（如果提供了 targets 参数）。
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

    # 在自然语言处理中，特殊 token 是在序列中具有特殊含义的标记，它们不同于常规的词语，通常用于在模型中传达特定的信息或执行特定的操作。其中两个常见的特殊 token 是 padding token 和 cls token。

    # Padding token（填充标记）： 在序列处理中，为了使所有序列具有相同的长度，通常会使用填充标记来将较短的序列填充到相同的长度。填充标记通常用于序列的末尾，并且具有一个特定的索引，以便模型能够识别它们并将其忽略。在训练过程中，填充标记不参与损失的计算。

    # CLS token（分类标记）： 在诸如BERT等预训练模型中，通常会在输入序列的开头添加一个特殊的分类标记（CLS token）。这个标记的输出用于整个序列的分类任务，如文本分类或序列标注。模型在处理序列时，将CLS token的输出作为整个序列的表示，并用它来进行分类任务的预测。

    def change_smi(self, input_ids, vocab_size, device, probability_change=0.5, probability_matrix=0.05):
        # probability_change_maxtrix (bs)
        # probability_matrix (bs,max_len)
        # (bs) bool
        # probability_change_maxtrix = torch.full((input_ids.shape[0],),probability_change)
        # probability_change_maxtrix = torch.bernoulli(probability_change_maxtrix).bool()  # 50%的更改,50%不变，0表示不改动,1表示改动,label应为1 - probability_change_maxtrix
        # probability_matrix = torch.full(input_ids.shape, probability_matrix) * probability_change_maxtrix.unsqueeze(-1)
        # masked_indices = torch.bernoulli(probability_matrix).bool()
        # # pad_token_id=0,cls_token_id=12,mask_token_id=14 不更改特殊字符
        # masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        # masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        # print(torch.sum(masked_indices, dim=-1))
        # print(1 - probability_change_maxtrix.long())
        # # 100% of the time, we replace masked input tokens with random word
        # indices_replace = torch.bernoulli(torch.full(input_ids.shape, 1.0)).bool() & masked_indices
        # random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        # input_ids[indices_replace] = random_words[indices_replace]
        '''
        #这个方法 change_smi 似乎是设计用来对表示为标记化 SMILES 字符串的输入序列进行扰动或修改的。以下是它的功能说明：

        #输入：
        #input_ids：表示为标记 ID 的输入序列。
        #vocab_size：词汇表的大小。
        #device：进行操作的设备。
        #probability_change：更改每个标记的概率。
        #probability_matrix：控制更改每个标记的可能性的概率矩阵。
        #功能：
        #它生成一个布尔矩阵 masked_indices，以确定基于 probability_matrix 哪些标记需要替换。某些标记，如填充标记或分类标记，不会被修改。
        #它创建另一个布尔张量 probability_change_maxtrix，根据 probability_change 确定每个序列是否会进行修改。
        #然后，它对输入序列应用更改：
        #根据 probability_change_maxtrix 和 masked_indices，随机替换标记为不同的标记 ID。
        #输出：修改后的 input_ids 张量，其中某些标记可能已被替换。一个张量，指示每个序列是否已被修改，允许跟踪更改。
        '''

        probability_change_maxtrix = torch.full((input_ids.shape[0],), probability_change)
        probability_change_maxtrix = torch.bernoulli(
            probability_change_maxtrix).bool()  # 50%的更改,50%不变，0表示不改动,1表示改动,label应为1 - probability_change_maxtrix

        # replace多个原子
        probability_matrix = torch.full(input_ids.shape, probability_matrix)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices[input_ids == self.tokenizer.pad_token_id] = 0.
        masked_indices[input_ids == self.tokenizer.cls_token_id] = 0.
        masked_indices = masked_indices * probability_change_maxtrix.unsqueeze(-1)  # 随机选择

        # # replace一个原子
        # masked_indices = torch.randn(input_ids.shape)
        # masked_indices[input_ids == self.tokenizer.pad_token_id] = 0.
        # masked_indices[input_ids == self.tokenizer.cls_token_id] = 0.
        # # 每行最大值取1，否则取0* probability_change_maxtrix.unsqueeze(-1)
        # masked_indices = (masked_indices == masked_indices.max(dim=1, keepdim=True)[0]).to(dtype=torch.int32).bool()
        # masked_indices = masked_indices * probability_change_maxtrix.unsqueeze(-1) # 随机选择

        indices_replace = torch.bernoulli(torch.full(input_ids.shape, 1.0)).bool() & masked_indices
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_replace] = random_words[indices_replace]

        return input_ids, (
                1 - (torch.sum(masked_indices, dim=-1).bool().int() ^ (1 - probability_change_maxtrix.long())) + (
                1 - probability_change_maxtrix.long()))

    def shufflePair(self, input_ids, images, image_false, vocab_size, device):
        '''

        这个方法 shufflePair 看起来是用来对输入的 SMILES 序列和图像进行打乱或修改的。下面是它的功能说明：

        输入：

        input_ids：表示为标记 ID 的输入序列。
        images：图像张量。
        image_false：替代图像张量。
        vocab_size：词汇表的大小。
        device：进行操作的设备。
        功能：

        它首先为每个样本生成一个标签，表示要执行的操作类型：
        0 表示不执行任何操作。
        1 表示对图像进行打乱。
        2 表示随机替换原子。
        3 表示随机删除原子。
        接下来，根据生成的标签，对输入序列和图像进行相应的操作：
        如果标签为 1，则将对应位置的图像替换为替代图像。
        如果标签为 2，则对对应位置的输入序列执行随机替换原子的操作。
        如果标签为 3，则对对应位置的输入序列执行随机删除原子的操作。
        最后，它将标签进行转换，以便表示哪些样本已被修改。
        输出：

        修改后的 input_ids 序列。
        修改后的图像张量。
        标签张量，指示每个样本进行了哪些修改。
        '''
        # 0 表示不动, 1 表示shuffle image , 2 表示randomReplaceAtoms , 3 表示randomDeleteAtoms
        labels = torch.randint(0, 4, (input_ids.shape[0],))

        # 1 表示shuffle image
        index_1 = torch.nonzero(labels == 1).squeeze(1)
        if index_1.shape[0] != 0:
            images[index_1] = image_false[index_1]

        # 表示randomReplaceAtoms
        index_2 = torch.nonzero(labels == 2).squeeze(1)
        if index_2.shape[0] != 0:
            input_ids[index_2] = self.randomReplaceAtoms(input_ids[index_2], vocab_size, device, probability_matrix=0.1)

        # 3 表示randomDeleteAtoms
        index_3 = torch.nonzero(labels == 3).squeeze(1)
        if index_3.shape[0] != 0:
            input_ids[index_3] = self.randomDeleteAtoms(input_ids[index_3])

        # label1,2,3,变成 0, 0 变成 1
        labels = 1 - labels.bool().float()

        return input_ids, images, labels

    def randomReplaceAtoms(self, input_ids, vocab_size, device, probability_matrix=0.1):
        '''
        randomReplaceAtoms 方法：

        输入：
        input_ids：表示为标记 ID 的输入序列。
        vocab_size：词汇表的大小。
        device：进行操作的设备。
        probability_matrix：一个标量，表示要替换原子的概率。

        功能：
        生成一个概率矩阵，确定要替换原子的位置。
        掩码特殊标记，如 <pad> 和 <cls>，以确保它们不会被替换。
        根据生成的掩码，确定要替换原子的位置。
        从词汇表中随机选择一个新的原子，并将其放置在指定的位置上。

        输出：
        修改后的 input_ids 序列。
        '''
        # replace多个原子
        probability_matrix = torch.full(input_ids.shape, probability_matrix)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices[input_ids == self.tokenizer.pad_token_id] = 0.  # 不替换 <pad>
        masked_indices[input_ids == self.tokenizer.cls_token_id] = 0.  # 不替换 <cls>
        indices_replace = masked_indices
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_replace] = random_words[indices_replace]
        return input_ids

    def randomDeleteAtoms(self, input_ids):
        '''
        randomDeleteAtoms 方法：

        输入：
        input_ids：表示为标记 ID 的输入序列。
        功能：
        将 <cls> 标记的原子分配给 0，表示不删除。
        将 <pad> 标记的原子分配给 1，表示可删除。
        计算每个标记的分数，并根据分数对标记进行排序。
        根据分数决定哪些原子要删除。
        删除标记，并确保不删除 <cls> 标记。
        输出：
        修改后的 input_ids 序列。
        '''
        pad = self.tokenizer.pad_token_id
        cls = self.tokenizer.cls_token_id
        max_len = input_ids.size(1)  # token length
        target_mask = input_ids.eq(pad)  # 填充为 1, (bs,max_len)
        target_score = input_ids.clone().float().uniform_()  # target_score
        target_score.masked_fill_(
            input_ids.eq(cls), 0.0  # <cls> 标识为 0
        )
        target_score.masked_fill_(target_mask, 1)  # <pad> 标识为 1
        target_score, target_rank = target_score.sort(1)  #
        target_length = target_mask.size(1) - target_mask.float().sum(  # 总长度
            1, keepdim=True
        )
        # do not delete <cls> (we assign 0 score for them)
        target_cutoff = (  # 目标长度
                1
                + (
                        (target_length - 1)
                        * target_score.new_zeros(target_score.size(0), 1).uniform_(0.7, 0.99)
                ).long()
        )
        target_cutoff = target_score.sort(1)[1] >= target_cutoff
        prev_target_tokens = (
            input_ids.gather(1, target_rank)
            .masked_fill_(target_cutoff, pad)
            .gather(1, target_rank.masked_fill_(target_cutoff, max_len).sort(1)[1])
        )
        return prev_target_tokens

    # def randomAddAtoms(self,input_ids):


def myclip_pretrain(_config):
    model = myCLIP(_config)
    return model


def one_hot_bonds(bonds):
    vocab_sizes = get_bond_feature_dims()
    one_hots = []
    for i in range(bonds.shape[1]):
        one_hots.append(F.one_hot(bonds[:, i], num_classes=vocab_sizes[i]).to(bonds.device))
    return torch.cat(one_hots, dim=1).float()


def one_hot_atoms(atoms):
    vocab_sizes = get_atom_feature_dims()
    one_hots = []
    for i in range(atoms.shape[1]):
        one_hots.append(F.one_hot(atoms[:, i], num_classes=vocab_sizes[i]).to(atoms.device))

    return torch.cat(one_hots, dim=1).float()


def combine_features(x, pos):
    """
    将节点特征和3D坐标数据结合起来。

    Args:
        x (torch.Tensor): 节点特征矩阵，维度为 [num_nodes, num_node_features]
        pos (torch.Tensor): 3D坐标矩阵，维度为 [num_nodes, 3]

    Returns:
        torch.Tensor: 融合后的特征矩阵，维度为 [num_nodes, num_node_features + 3]
    """
    return torch.cat([x, pos], dim=1)


def smi2_3Dcoords(smi, cnt=1):
    mol = Chem.MolFromSmiles(smi)

    mol = AllChem.AddHs(mol)

    coordinate_list = []
    for seed in range(cnt):
        try:
            res = AllChem.EmbedMolecule(mol, randomSeed=seed)
            if res == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol)
                    coordinates = mol.GetConformer().GetPositions()
                except:
                    print("Failed to generate 3D, replace with 2D")
                    coordinates = smi2_2Dcoords(smi)
            elif res == -1:
                mol_tmp = Chem.MolFromSmiles(smi)
                AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000, randomSeed=seed)
                mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
                try:
                    AllChem.MMFFOptimizeMolecule(mol_tmp)
                    coordinates = mol_tmp.GetConformer().GetPositions()
                except:
                    print("Failed to generate 3D, replace with 2D")
                    coordinates = smi2_2Dcoords(smi)
        except:
            print("Failed to generate 3D, replace with 2D")
            coordinates = smi2_2Dcoords(smi)

        assert len(mol.GetAtoms()) == len(coordinates), "3D coordinates shape is not align with {}".format(smi)
        coordinate_list.append(coordinates.astype(np.float32))

    return coordinate_list


def process_data1(smiles_list, smiles2graph, isomorphic_core):
    batch_data_list = []  # 用于存储每个分子的Data对象列表

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        # mol = AllChem.AddHs(mol)  # 添加氢原子

        pos = smi2_3Dcoords(smiles)[0]  # 获取3D坐标

        data = Data()
        graph = smiles2graph(mol)  # 根据SMILES字符串获取图表示

        # 填充Data对象的属性
        assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
        assert len(graph["node_feat"]) == graph["num_nodes"]

        data.__num_nodes__ = int(graph["num_nodes"])
        data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
        data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
        data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
        data.y = None

        data.ring_mask = torch.from_numpy(graph["ring_mask"]).to(torch.bool)
        data.ring_index = torch.from_numpy(graph["ring_index"]).to(torch.int64)
        data.nf_node = np.array(graph["nf_node"])
        data.nf_ring = np.array(graph["nf_ring"])
        data.num_rings = int(graph["num_rings"])
        data.n_edges = int(graph["n_edges"])
        data.n_nodes = int(graph["n_nodes"])
        data.n_nfs = int(graph["n_nfs"])

        data.pos = torch.from_numpy(pos).to(torch.float)
        data.isomorphisms = isomorphic_core(mol)
        # print(f'isomorphisms',data.isomorphisms)
        # 将当前分子的Data对象添加到列表中
        batch_data_list.append(data)

    # 使用Batch.from_data_list将列表中的所有Data对象合并为一个批次
    # for data in batch_data_list:
    #    print(f"data.x:",data.x.size())
    #    print(f"data.pos:",data.pos.size())
    data_batch = Batch.from_data_list(batch_data_list)

    return data_batch


def process_data(smiles_list, poss_list, smiles2graph, isomorphic_core):
    batch_data_list = []  # 用于存储每个分子的Data对象列表

    for smiles, pos in zip(smiles_list, poss_list):
        mol = Chem.MolFromSmiles(smiles)

        # mol = AllChem.AddHs(mol)  # 添加氢原子

        pos_list_of_lists = ast.literal_eval(pos)
        pos = np.array(pos_list_of_lists)  # 将位置列表转换为NumPy数组
        # pos = smi2_3Dcoords(smiles)[0]  # 获取3D坐标

        data = Data()
        graph = smiles2graph(mol)  # 根据SMILES字符串获取图表示
        # graph = smiles2graphwithface(mol)

        # 填充Data对象的属性
        assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
        assert len(graph["node_feat"]) == graph["num_nodes"]

        # data.__num_nodes__ = int(graph["num_nodes"])
        # data.edge_index = np.array(graph["edge_index"])
        # data.edge_attr = np.array(graph["edge_feat"])
        # data.x = np.array(graph["node_feat"])
        # data.y = None

        # data.ring_mask = np.array(graph["ring_mask"])
        # data.ring_index = np.array(graph["ring_index"])
        # data.nf_node = np.array(graph["nf_node"])
        # data.nf_ring = np.array(graph["nf_ring"])
        # data.num_rings = int(graph["num_rings"])
        # data.n_edges = int(graph["n_edges"])
        # data.n_nodes = int(graph["n_nodes"])
        # data.n_nfs = int(graph["n_nfs"])
        # data.graph = graph
        # data.pos = pos
        # batch_data_list.append(data)

        data.__num_nodes__ = int(graph["num_nodes"])

        data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
        data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
        data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
        # print(data.x.shape)
        #        if data.x.size(0) != len(pos):
        #             print(pos[0])
        #             pos = smi2_3Dcoords(smiles)[0]  # 获取3D坐标
        #             print(smiles)
        #             print(data.x.size(0))
        #
        #             print("################################################################")
        data.y = None

        data.ring_mask = torch.from_numpy(graph["ring_mask"]).to(torch.bool)
        data.ring_index = torch.from_numpy(graph["ring_index"]).to(torch.int64)
        data.nf_node = np.array(graph["nf_node"])
        data.nf_ring = np.array(graph["nf_ring"])
        data.num_rings = int(graph["num_rings"])
        data.n_edges = int(graph["n_edges"])
        data.n_nodes = int(graph["n_nodes"])
        data.n_nfs = int(graph["n_nfs"])
        if data.x.shape[0] != len(pos):
            pos = smi2_3Dcoords(smiles)[0]


        data.pos = torch.from_numpy(pos).to(torch.float)
        # print(data.pos.shape)
        data.isomorphisms = isomorphic_core(mol)
        # print(f'isomorphisms',data.isomorphisms)
        # 将当前分子的Data对象添加到列表中
        batch_data_list.append(data)

    # 使用Batch.from_data_list将列表中的所有Data对象合并为一个批次

    data_batch = Batch.from_data_list(batch_data_list)
    # for data in data_batch:
    # print(f"data:",data)
    # print(f"data.num_graphs:",data.num_graphs)
    return data_batch


# 该函数处理后的batch_data_list为Data对象列表，其中的所有属性均为列表类型


def smi2_3Dcoords(smi, cnt=1):
    mol = Chem.MolFromSmiles(smi)

    # mol = AllChem.AddHs(mol)

    coordinate_list = []
    for seed in range(cnt):
        try:
            res = AllChem.EmbedMolecule(mol, randomSeed=seed)
            if res == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol)
                    coordinates = mol.GetConformer().GetPositions()
                except:
                    print("Failed to generate 3D, replace with 2D")
                    coordinates = smi2_2Dcoords(smi)
            elif res == -1:
                mol_tmp = Chem.MolFromSmiles(smi)
                AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000, randomSeed=seed)
                # mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
                try:
                    AllChem.MMFFOptimizeMolecule(mol_tmp)
                    coordinates = mol_tmp.GetConformer().GetPositions()
                except:
                    print("Failed to generate 3D, replace with 2D")
                    coordinates = smi2_2Dcoords(smi)
        except:
            print("Failed to generate 3D, replace with 2D")
            coordinates = smi2_2Dcoords(smi)

        assert len(mol.GetAtoms()) == len(coordinates), "3D coordinates shape is not align with {}".format(smi)
        coordinate_list.append(coordinates.astype(np.float32))
    return coordinate_list


def smi2_2Dcoords(smi):
    mol = Chem.MolFromSmiles(smi)
    # mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    assert len(mol.GetAtoms()) == len(coordinates), "2D coordinates shape is not align with {}".format(smi)
    return coordinates