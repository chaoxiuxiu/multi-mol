from sacred import Experiment
import math
ex = Experiment("myCLIP",save_git_info=False)


def _loss_names(d):
    ret = {
        # pre-task
        "itm": 0,
        "mlm": 0,
        "mpp": 0,
        "itc": 0,
        "gcl": 0,
        "m2d": 0,
        "c2m": 0,
        "m2c": 0,
        "gtm": 0,
        "fpp": 0,
        "jem": 0,
        # downstream task
        "imcap": 0,
        "MoleculeNet_classify": 0,
        'MoleculeNet_regress': 0,
        'ocsr': 0,
        "ocsr_finturn": 0,
        'inter': 0,
        'auroc': 0
    }
    ret.update(d)
    return ret

@ex.config
def config():
    exp_name = "pretrain"
    seed = 0
    datasets = ["pretrain"]
    #loss_names = _loss_names({ "mlm": 1 , "fpp": 1,'itm': 1})
    loss_names = _loss_names({"gcl": 1 , "mlm": 1 , 'itm': 1,"m2d": 1,"c2m": 1,"m2c": 1,'gtm': 0, 'fpp':0, 'jem':0})

    # loss_names = _loss_names({"itm": 1})
    # loss_names = _loss_names({"mpp": 1})
    # loss_names = _loss_names({"fpp": 1})
    # loss_names = _loss_names({"mlm": 1,"itm": 1})
    batch_size = 2048#2048  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # Image setting
    max_image_len = -1
    image_size = 224
    patch_size = 16
    resolution_before = 224
    encoder_width = 256
    mask_ratio = 0.25

    # Text Setting
    max_smiles_len = 202
    tokenizer = "../mae-main/ouput_dir75/checkpoint-100.pth"
    vocab_size = 591
    mlm_probability = 0.15

    # Transformer Setting
    num_top_layer = 3
    input_image_embed_size = 768
    input_smiles_embed_size = 768
    hidden_size = 768
    num_heads = 12
    num_layers = 6
    mlp_ratio = 4
    drop_rate = 0.1

    # Optimizer Setting
    optim_type = "adamw"#"adamw"
    learning_rate = 1.5e-4#1.5e-4
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 60#60
    max_steps = 10000#10000
    warmup_steps = 0.1#0.1
    end_lr = 0
    lr_mult_head = 5  # multiply lr for downstream heads
    lr_mult_cross_modal = 5  # multiply lr for the cross-modal module

    # Downstream Setting

    # PL Trainer Setting
    resume_from = None# '/data1/zx/myCLIP/cpkt_path/p1-epoch=14-global_step=0-025-v1.ckpt'
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False

    # below params varies with the environment
    #data_root = '/data1/zx/data/ScaffoldCount10' # './data/dataset_pyarray'
    data_root = '../data/ScaffoldCount1000' # './data/dataset_pyarray'
    #data_root = '/root/autodl-tmp/myCLIP/data/dataset_pyarray'
    log_dir = "result"
    checkpoint_save_path = "./cpkt_path"
    mode = 'max'
    per_gpu_batchsize = 25   # you should define this manually with per_gpu_batch_size=#
    num_gpus = [0]
    num_nodes = 1
     # path to load pretraining checkpoint
    load_path =""
    num_workers = 6
    precision = 32
    is_pretrain = True
    imbsampler = False  # 处理样本均衡问题，是否对数据进行imbalance sampler
    log_project = 'myCLIP'

@ex.named_config
def task_finetune_MoleculeNet_classify():
    exp_name = "finetune_classifier"
    datasets = ['MoleculeNet_classify']     # 直接映射到分类数据加载模块
    loss_names = _loss_names({"MoleculeNet_classify": 1})

    # batch_size = [32,64,128]
    # max_steps = [1000,1500,2000,2500] # 2500
    # learning_rate = [5e-6,1e-5,5e-5]

    warmup_steps = 0.1#0.1
    is_pretrain = False
    load_path = ''
    per_gpu_batchsize = 32 #16
    seed = 0
    test_only = False
    decay_power = "cosine" #1
    mode = 'max'   # 监测验证集最大值时的checkpoint,

############################## classify ###################################
@ex.named_config
def task_finetune_MoleculeNet_classify_CYPE():
    data_root = './data/downstreamDataset/ADMETlab_data/CycPeptMPDB_Peptide_3d'
    checkpoint_save_path = './cpkt_path/ADMETlab_data1/CycPeptMPDB_Peptide'
    load_path = ""#
    log_dir = 'downstream_result/ADMETlab_data/CycPeptMPDB_Peptide'
    log_project = 'cype'
    imbsampler = False # 处理样本不均问题


@ex.named_config
def task_finetune_MoleculeNet_classify_CYPE_KANO():
    data_root = './data/downstreamDataset/ADMETlab_data/labeled_Peptide5'
    checkpoint_save_path = './cpkt_path/ADMETlab_data1/CycPeptMPDB_Peptide5'
    load_path = ""
    log_dir = 'downstream_result/ADMETlab_data/CycPeptMPDB_Peptide'
    log_project = 'cype'
    imbsampler = False # 处理样本不均问题


@ex.named_config
def task_finetune_MoleculeNet_classify_BBBP():
    data_root = './data/downstreamDataset/ADMETlab_data/BBBP'
    checkpoint_save_path = "./cpkt_path/ADMETlab_data1/BBBP"
    log_dir = 'downstream_result/MoleculeNet/BBBP'


@ex.named_config
def task_finetune_MoleculeNet_classify_BACE():
    # '/data1/zx/data/task_data/bace_c/images' # './data/downstreamDataset/MoleculeNet/BACE'
    # '/data1/zx/data/task_data_nop/bace_c/images'
    data_root = './data/downstreamDataset/ADMETlab_data/BACE'
    checkpoint_save_path = './cpkt_path/ADMETlab_data1/BACE'
    log_dir = 'downstream_result/MoleculeNet/BACE'
    seed = 0

@ex.named_config
def task_finetune_MoleculeNet_classify_Hiv():
    data_root = './data/downstreamDataset/ADMETlab_data/Hiv'
    checkpoint_save_path = './cpkt_path/ADMETlab_data1/Hiv'
    log_dir = 'downstream_result/MoleculeNet/Hiv'
    per_gpu_batchsize = 64
    #optim_type = "rmsprop"
    imbsampler = True # 处理样本不均问题

@ex.named_config
def task_finetune_MoleculeNet_classify_Tox21():
    data_root = './data/downstreamDataset/MoleculeNet/Tox21'
    checkpoint_save_path = './cpkt_path/Molecule1/Tox21'
    log_dir = 'downstream_result/MoleculeNet/Tox21'
    imbsampler = False  # 处理样本不均问题
    batch_size = 64
    max_steps = 2500
    learning_rate = 5e-5
    seed = 10




@ex.named_config
def task_finetune_MoleculeNet_classify_ClinTox():
    data_root = './data/downstreamDataset/MoleculeNet/ClinTox'
    checkpoint_save_path = './cpkt_path/Molecule/ClinTox'
    log_dir = 'downstream_result/MoleculeNet/ClinTox'
    imbsampler = True  # 处理样本不均问题

@ex.named_config
def task_finetune_MoleculeNet_classify_ClinTox0():
    data_root = './data/downstreamDataset/MoleculeNet/clintox_0'
    checkpoint_save_path = './cpkt_path/Molecule/clintox_0'
    log_dir = 'downstream_result/MoleculeNet/clintox_0'
    imbsampler = True  # 处理样本不均问题

@ex.named_config
def task_finetune_MoleculeNet_classify_ClinTox1():
    data_root = './data/downstreamDataset/MoleculeNet/clintox_1'
    checkpoint_save_path = './cpkt_path/Molecule/clintox_1'
    log_dir = 'downstream_result/MoleculeNet/clintox_1'
    imbsampler = True  # 处理样本不均问题


############################################################
@ex.named_config
def task_finetune_MoleculeNet_classify_HIA():
    data_root = './data/downstreamDataset/ADMETlab_data/HIA'
    checkpoint_save_path = './cpkt_path/ADMETlab_data1/HIA'
    log_dir = 'downstream_result/ADMETlab_data/HIA'
    log_project = 'hia'
    imbsampler = False  # 处理样本不均问题

@ex.named_config
def task_finetune_MoleculeNet_classify_HIA_imageMol():
    data_root = './data/downstreamDataset/ADMETlab_data/HIA'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/HIA_imageMol'
    log_dir = 'downstream_result/ADMETlab_data/HIA_imageMol'
    log_project = 'HIA_imageMol'
    imbsampler = True  # 处理样本不均问题

@ex.named_config
def task_finetune_MoleculeNet_classify_HIA_chemberta():
    data_root = './data/downstreamDataset/ADMETlab_data/HIA'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/HIA_chemberta'
    log_dir = 'downstream_result/ADMETlab_data/HIA_chemberta'
    log_project = 'HIA_chemberta'
    imbsampler = True  # 处理样本不均问题

@ex.named_config
def task_finetune_MoleculeNet_classify_HIA_vit():
    data_root = './data/downstreamDataset/ADMETlab_data/HIA'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/HIA_vit'
    log_dir = 'downstream_result/ADMETlab_data/HIA_vit'
    log_project = 'HIA_vit'
    imbsampler = True  # 处理样本不均问题
############################################################


############################################################
@ex.named_config
def task_finetune_MoleculeNet_classify_pgb_sub():
    data_root = './data/downstreamDataset/ADMETlab_data/Pgp-sub'
    checkpoint_save_path = './cpkt_path/ADMETlab_data1/pgb_sub'
    log_dir = 'downstream_result/ADMETlab_data/pgb_sub'
    log_project = 'pgb_sub'
    imbsampler = False#True  # 处理样本不均问题

@ex.named_config
def task_finetune_MoleculeNet_classify_pgb_sub_imageMol():
    data_root = './data/downstreamDataset/ADMETlab_data/pgb_sub'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/pgb_sub_imageMol'
    log_dir = 'downstream_result/ADMETlab_data/pgb_sub_imageMol'
    log_project = 'pgb_sub_imageMol'
    imbsampler = True  # 处理样本不均问题
    per_gpu_batchsize = 64

@ex.named_config
def task_finetune_MoleculeNet_classify_pgb_sub_chemberta():
    data_root = './data/downstreamDataset/ADMETlab_data/pgb_sub'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/pgb_sub_chemberta'
    log_dir = 'downstream_result/ADMETlab_data/pgb_sub_chemberta'
    log_project = 'pgb_sub_chemberta'
    imbsampler = True  # 处理样本不均问题

@ex.named_config
def task_finetune_MoleculeNet_classify_pgb_sub_vit():
    data_root = './data/downstreamDataset/ADMETlab_data/pgb_sub'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/pgb_sub_vit'
    log_dir = 'downstream_result/ADMETlab_data/pgb_sub_vit'
    log_project = 'pgb_sub_vit'
    imbsampler = True  # 处理样本不均问题
############################################################

@ex.named_config
def task_finetune_MoleculeNet_classify_F20():
    data_root = './data/downstreamDataset/ADMETlab_data/F20'
    checkpoint_save_path = './cpkt_path/ADMETlab_data1/F20'
    log_dir = 'downstream_result/ADMETlab_data/F20'
    log_project = 'F20'

@ex.named_config
def task_finetune_MoleculeNet_classify_F20_imageMol():
    data_root = './data/downstreamDataset/ADMETlab_data/F20'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/F20_imageMol'
    log_dir = 'downstream_result/ADMETlab_data/F20_imageMol'
    log_project = 'F20_imageMol'

@ex.named_config
def task_finetune_MoleculeNet_classify_F20_chemberta():
    data_root = './data/downstreamDataset/ADMETlab_data/F20'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/F20_chemberta'
    log_dir = 'downstream_result/ADMETlab_data/F20_chemberta'
    log_project = 'F20_chemberta'

@ex.named_config
def task_finetune_MoleculeNet_classify_F20_vit():
    data_root = './data/downstreamDataset/ADMETlab_data/F20'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/F20_vit'
    log_dir = 'downstream_result/ADMETlab_data/F20_vit'
    log_project = 'F20_vit'
############################################################

@ex.named_config
def task_finetune_MoleculeNet_classify_F30():
    data_root = './data/downstreamDataset/ADMETlab_data/F30'
    checkpoint_save_path = './cpkt_path/ADMETlab_data1/F30'
    log_dir = 'downstream_result/ADMETlab_data/F30'
    #optim_type = "adam"#"adamw"
    warmup_steps=0.3#0.1
    log_project = 'F30'



@ex.named_config
def task_finetune_MoleculeNet_classify_F30_imageMol():
    data_root = './data/downstreamDataset/ADMETlab_data/F30'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/F30_imageMol'
    log_dir = 'downstream_result/ADMETlab_data/F30_imageMol'
    log_project = 'F30_imageMol'
    imbsampler = True

@ex.named_config
def task_finetune_MoleculeNet_classify_F30_chemberta():
    data_root = './data/downstreamDataset/ADMETlab_data/F30'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/F30_chemberta'
    log_dir = 'downstream_result/ADMETlab_data/F30_chemberta'
    log_project = 'F30_chemberta'
    imbsampler = True

@ex.named_config
def task_finetune_MoleculeNet_classify_F30_vit():
    data_root = './data/downstreamDataset/ADMETlab_data/F30'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/F30_vit'
    log_dir = 'downstream_result/ADMETlab_data/F30_vit'
    log_project = 'F30_vit'
    imbsampler = True
############################################################

@ex.named_config
def task_finetune_MoleculeNet_classify_FDAMDD():
    data_root = './data/downstreamDataset/ADMETlab_data/FDAMDD'
    checkpoint_save_path = './cpkt_path/ADMETlab_data1/FDAMDD'
    log_dir = 'downstream_result/ADMETlab_data/FDAMDD'
    log_project = 'FDAMDD'

@ex.named_config
def task_finetune_MoleculeNet_classify_FDAMDD_imageMol():
    data_root = './data/downstreamDataset/ADMETlab_data/FDAMDD'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/FDAMDD_imageMol'
    log_dir = 'downstream_result/ADMETlab_data/FDAMDD_imageMol'
    log_project = 'FDAMDD_imageMol'

@ex.named_config
def task_finetune_MoleculeNet_classify_FDAMDD_chemberta():
    data_root = './data/downstreamDataset/ADMETlab_data/FDAMDD'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/FDAMDD_chemberta'
    log_dir = 'downstream_result/ADMETlab_data/FDAMDD_chemberta'
    log_project = 'FDAMDD_chemberta'

@ex.named_config
def task_finetune_MoleculeNet_classify_FDAMDD_vit():
    data_root = './data/downstreamDataset/ADMETlab_data/FDAMDD'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/FDAMDD_vit'
    log_dir = 'downstream_result/ADMETlab_data/FDAMDD_vit'
    log_project = 'FDAMDD_vit'
############################################################

@ex.named_config
def task_finetune_MoleculeNet_classify_CYP1A2():
    data_root = './data/downstreamDataset/ADMETlab_data/CYP1A2_sub'
    checkpoint_save_path = './cpkt_path/ADMETlab_data1/CYP1A2_sub'
    log_dir = 'downstream_result/ADMETlab_data/CYP1A2_sub'
    log_project = 'CYP1A2'
    imbsampler = True

@ex.named_config
def task_finetune_MoleculeNet_classify_CYP1A2_imageMol():
    data_root = './data/downstreamDataset/ADMETlab_data/CYP1A2_sub'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/CYP1A2_sub_imageMol'
    log_dir = 'downstream_result/ADMETlab_data/CYP1A2_sub_imageMol'
    log_project = 'CYP1A2_imageMol'
    imbsampler = True

@ex.named_config
def task_finetune_MoleculeNet_classify_CYP1A2_chemberta():
    data_root = './data/downstreamDataset/ADMETlab_data/CYP1A2_sub'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/CYP1A2_sub_chemberta'
    log_dir = 'downstream_result/ADMETlab_data/CYP1A2_sub_chemberta'
    log_project = 'CYP1A2_chemberta'
    imbsampler = True

@ex.named_config
def task_finetune_MoleculeNet_classify_CYP1A2_vit():
    data_root = './data/downstreamDataset/ADMETlab_data/CYP1A2_sub'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/CYP1A2_sub_vit'
    log_dir = 'downstream_result/ADMETlab_data/CYP1A2_sub_vit'
    log_project = 'CYP1A2_vit'
    imbsampler = True

############################################################

@ex.named_config
def task_finetune_MoleculeNet_classify_CYP2C19():
    data_root = './data/downstreamDataset/ADMETlab_data/CYP2C19_sub'
    checkpoint_save_path = './cpkt_path/ADMETlab_data1/CYP2C19_sub'
    log_dir = 'downstream_result/ADMETlab_data/CYP2C19_sub'
    log_project = 'CYP2C19'
    per_gpu_batchsize = 16
    imbsampler =True

@ex.named_config
def task_finetune_MoleculeNet_classify_CYP2C19_imageMol():
    data_root = './data/downstreamDataset/ADMETlab_data/CYP2C19_sub'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/CYP2C19_sub_imageMol'
    log_dir = 'downstream_result/ADMETlab_data/CYP2C19_sub_imageMol'
    log_project = 'CYP2C19_imageMol'

@ex.named_config
def task_finetune_MoleculeNet_classify_CYP2C19_chemberta():
    data_root = './data/downstreamDataset/ADMETlab_data/CYP2C19_sub'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/CYP2C19_sub_chemberta'
    log_dir = 'downstream_result/ADMETlab_data/CYP2C19_sub_chemberta'
    log_project = 'CYP2C19_chemberta'

@ex.named_config
def task_finetune_MoleculeNet_classify_CYP2C19_vit():
    data_root = './data/downstreamDataset/ADMETlab_data/CYP2C19_sub'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/CYP2C19_sub_vit'
    log_dir = 'downstream_result/ADMETlab_data/CYP2C19_sub_vit'
    log_project = 'CYP2C19_vit'

############################################################

@ex.named_config
def task_finetune_MoleculeNet_classify_CYP2C9():
    data_root = './data/downstreamDataset/ADMETlab_data/CYP2C9_sub'
    checkpoint_save_path = './cpkt_path/ADMETlab_data1/CYP2C9_sub'
    log_dir = 'downstream_result/ADMETlab_data/CYP2C9_sub'
    #warmup_steps = 0.3#0.1
    log_project = 'CYP2C9'
    
    imbsampler = True

@ex.named_config
def task_finetune_MoleculeNet_classify_CYP2C9_imageMol():
    data_root = './data/downstreamDataset/ADMETlab_data/CYP2C9_sub'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/CYP2C9_sub_imageMol'
    log_dir = 'downstream_result/ADMETlab_data/CYP2C9_sub_imageMol'
    log_project = 'CYP2C9_imageMol'
    imbsampler = True

@ex.named_config
def task_finetune_MoleculeNet_classify_CYP2C9_chemberta():
    data_root = './data/downstreamDataset/ADMETlab_data/CYP2C9_sub'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/CYP2C9_sub_chemberta'
    log_dir = 'downstream_result/ADMETlab_data/CYP2C9_sub_chemberta'
    log_project = 'CYP2C9_chemberta'
    imbsampler = True

@ex.named_config
def task_finetune_MoleculeNet_classify_CYP2C9_vit():
    data_root = './data/downstreamDataset/ADMETlab_data/CYP2C9_sub'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/CYP2C9_sub_vit'
    log_dir = 'downstream_result/ADMETlab_data/CYP2C9_sub_vit'
    log_project = 'CYP2C9_vit'
    imbsampler = True
############################################################

@ex.named_config
def task_finetune_MoleculeNet_classify_CYP2D6():
    data_root = './data/downstreamDataset/ADMETlab_data/CYP2D6_sub'
    checkpoint_save_path = './cpkt_path/ADMETlab_data1/CYP2D6_sub'
    log_dir = 'downstream_result/ADMETlab_data/CYP2D6_sub'
    #optim_type = "adam"#"adamw"
    log_project = 'CYP2D6'

@ex.named_config
def task_finetune_MoleculeNet_classify_CYP2D6_imageMol():
    data_root = './data/downstreamDataset/ADMETlab_data/CYP2D6_sub'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/CYP2D6_sub_imageMol'
    log_dir = 'downstream_result/ADMETlab_data/CYP2D6_sub_imageMol'
    log_project = 'CYP2D6_imageMol'

@ex.named_config
def task_finetune_MoleculeNet_classify_CYP2D6_chemberta():
    data_root = './data/downstreamDataset/ADMETlab_data/CYP2D6_sub'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/CYP2D6_sub_chemberta'
    log_dir = 'downstream_result/ADMETlab_data/CYP2D6_sub_chemberta'
    log_project = 'CYP2D6_chemberta'
    imbsampler = True

@ex.named_config
def task_finetune_MoleculeNet_classify_CYP2D6_vit():
    data_root = './data/downstreamDataset/ADMETlab_data/CYP2D6_sub'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/CYP2D6_sub_vit'
    log_dir = 'downstream_result/ADMETlab_data/CYP2D6_sub_vit'
    log_project = 'CYP2D6_vit'
    imbsampler = True

############################################################

@ex.named_config
def task_finetune_MoleculeNet_classify_CYP3A4():
    data_root = './data/downstreamDataset/ADMETlab_data/CYP3A4_sub'
    checkpoint_save_path = './cpkt_path/ADMETlab_data1/CYP3A4_sub'

    log_dir = 'downstream_result/ADMETlab_data/CYP3A4_sub'
    #warmup_steps=0.3#0.1
    log_project = 'CYP3A4'

@ex.named_config
def task_finetune_MoleculeNet_classify_CYP3A4_imageMol():
    data_root = './data/downstreamDataset/ADMETlab_data/CYP3A4_sub'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/CYP3A4_sub_imageMol'
    log_dir = 'downstream_result/ADMETlab_data/CYP3A4_sub_imageMol'
    log_project = 'CYP3A4_imageMol'

@ex.named_config
def task_finetune_MoleculeNet_classify_CYP3A4_chemberta():
    data_root = './data/downstreamDataset/ADMETlab_data/CYP3A4_sub'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/CYP3A4_sub_chemberta'
    log_dir = 'downstream_result/ADMETlab_data/CYP3A4_sub_chemberta'
    log_project = 'CYP3A4_chemberta'

@ex.named_config
def task_finetune_MoleculeNet_classify_CYP3A4_vit():
    data_root = './data/downstreamDataset/ADMETlab_data/CYP3A4_sub'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/CYP3A4_sub_vit'
    log_dir = 'downstream_result/ADMETlab_data/CYP3A4_sub_vit'
    log_project = 'CYP3A4_vit'
############################################################

@ex.named_config
def task_finetune_MoleculeNet_classify_T12():
    data_root = './data/downstreamDataset/ADMETlab_data/T12'
    checkpoint_save_path = './cpkt_path/ADMETlab_data1/T12'

    log_dir = 'downstream_result/ADMETlab_data/T12'
    per_gpu_batchsize = 32
    #warmup_steps = 0.3#0.1
    log_project = 'T12'

@ex.named_config
def task_finetune_MoleculeNet_classify_T12_imageMol():
    data_root = './data/downstreamDataset/ADMETlab_data/T12'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/T12_imageMol'
    log_dir = 'downstream_result/ADMETlab_data/T12_imageMol'
    log_project = 'T12_imageMol'

@ex.named_config
def task_finetune_MoleculeNet_classify_T12_chemberta():
    data_root = './data/downstreamDataset/ADMETlab_data/T12'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/T12_chemberta'
    log_dir = 'downstream_result/ADMETlab_data/T12_chemberta'
    log_project = 'T12_chemberta'

@ex.named_config
def task_finetune_MoleculeNet_classify_T12_vit():
    data_root = './data/downstreamDataset/ADMETlab_data/T12'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/T12_vit'
    log_dir = 'downstream_result/ADMETlab_data/T12_vit'
    log_project = 'T12_vit'
############################################################
@ex.named_config
def task_finetune_MoleculeNet_classify_DILI():
    data_root = './data/downstreamDataset/ADMETlab_data/DILI'
    checkpoint_save_path = './cpkt_path/ADMETlab_data1/DILI'

    log_dir = 'downstream_result/ADMETlab_data/DILI'
    log_project = 'DILI'
    imbsampler = True

@ex.named_config
def task_finetune_MoleculeNet_classify_DILI_imageMol():
    data_root = './data/downstreamDataset/ADMETlab_data/DILI'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/DILI_imageMol'
    log_dir = 'downstream_result/ADMETlab_data/DILI_imageMol'
    log_project = 'DILI_imageMol'
    imbsampler = True

@ex.named_config
def task_finetune_MoleculeNet_classify_DILI_chemberta():
    data_root = './data/downstreamDataset/ADMETlab_data/DILI'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/DILI_chemberta'
    log_dir = 'downstream_result/ADMETlab_data/DILI_chemberta'
    log_project = 'DILI_chemberta'
    imbsampler = True

@ex.named_config
def task_finetune_MoleculeNet_classify_DILI_vit():
    data_root = './data/downstreamDataset/ADMETlab_data/DILI'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/DILI_vit'
    log_dir = 'downstream_result/ADMETlab_data/DILI_vit'
    log_project = 'DILI_vit'
    imbsampler = True
############################################################
@ex.named_config
def task_finetune_MoleculeNet_classify_SkinSen():
    data_root = './data/downstreamDataset/ADMETlab_data/SkinSen'
    checkpoint_save_path = './cpkt_path/ADMETlab_data1/SkinSen'

    log_dir = 'downstream_result/ADMETlab_data/SkinSen'
    warmup_steps = 0.4#0.1
    log_project = 'SkinSen'
    imbsampler = True

@ex.named_config
def task_finetune_MoleculeNet_classify_SkinSen_imageMol():
    data_root = './data/downstreamDataset/ADMETlab_data/SkinSen'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/SkinSen_imageMol'
    log_dir = 'downstream_result/ADMETlab_data/SkinSen_imageMol'
    log_project = 'SkinSen_imageMol'
    imbsampler = False

@ex.named_config
def task_finetune_MoleculeNet_classify_SkinSen_chemberta():
    data_root = './data/downstreamDataset/ADMETlab_data/SkinSen'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/SkinSen_chemberta'
    log_dir = 'downstream_result/ADMETlab_data/SkinSen_chemberta'
    log_project = 'SkinSen_chemberta'
    imbsampler = True

@ex.named_config
def task_finetune_MoleculeNet_classify_SkinSen_vit():
    data_root = './data/downstreamDataset/ADMETlab_data/SkinSen'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/SkinSen_vit'
    log_dir = 'downstream_result/ADMETlab_data/SkinSen_vit'
    log_project = 'SkinSen_vit'
    imbsampler = True
############################################################

@ex.named_config
def task_finetune_MoleculeNet_classify_Carcinogenicity():
    data_root = './data/downstreamDataset/ADMETlab_data/Carcinogenicity'
    checkpoint_save_path = './cpkt_path/ADMETlab_data1/Carcinogenicity'

    log_dir = 'downstream_result/ADMETlab_data/Carcinogenicity'
    log_project = 'Carcinogenicity'

@ex.named_config
def task_finetune_MoleculeNet_classify_Carcinogenicity_imageMol():
    data_root = './data/downstreamDataset/ADMETlab_data/Carcinogenicity'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/Carcinogenicity_imageMol'
    log_dir = 'downstream_result/ADMETlab_data/Carcinogenicity_imageMol'
    log_project = 'Carcinogenicity_imageMol'

############################################################
@ex.named_config
def task_finetune_MoleculeNet_classify_Respiratory():
    data_root = './data/downstreamDataset/ADMETlab_data/Respiratory'
    checkpoint_save_path = './cpkt_path/ADMETlab_data1/Respiratory'

    log_dir = 'downstream_result/ADMETlab_data/Respiratory'
    per_gpu_batchsize = 32
    log_project = 'Respiratory'

@ex.named_config
def task_finetune_MoleculeNet_classify_Respiratory_imageMol():
    data_root = './data/downstreamDataset/ADMETlab_data/Respiratory'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/Respiratory_imageMol'
    log_dir = 'downstream_result/ADMETlab_data/Respiratory_imageMol'
    log_project = 'Respiratory_imageMol'

@ex.named_config
def task_finetune_MoleculeNet_classify_Respiratory_chemberta():
    data_root = './data/downstreamDataset/ADMETlab_data/Respiratory'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/Respiratory_chemberta'
    log_dir = 'downstream_result/ADMETlab_data/Respiratory_chemberta'
    log_project = 'Respiratory_chemberta'

@ex.named_config
def task_finetune_MoleculeNet_classify_Respiratory_vit():
    data_root = './data/downstreamDataset/ADMETlab_data/Respiratory'
    checkpoint_save_path = './cpkt_path/ADMETlab_data/Respiratory_vit'
    log_dir = 'downstream_result/ADMETlab_data/Respiratory_vit'
    log_project = 'Respiratory_vit'
###########################################################################



############################## regress ####################################
@ex.named_config
def task_finetune_MoleculeNet_regress():
    exp_name = "finetune_regress"
    datasets = ['MoleculeNet_regress']   # 直接映射到回归数据加载模块
    loss_names = _loss_names({"MoleculeNet_regress": 1})

    is_pretrain = False
    warmup_steps = 0.1
    load_path = ''
    per_gpu_batchsize = 2
    mode = 'min'   # 监测验证集最小值时的checkpoint,


############################## classify ###################################
@ex.named_config
def task_finetune_MoleculeNet_regress_CYPE():
    data_root = './data/downstreamDataset/ADMETlab_data/CYPE_regress1'
    #data_root = './data/downstreamDataset/ADMETlab_data/CycPeptMPDB_Peptide3'
    checkpoint_save_path = './cpkt_path/ADMETlab_data1/CYPE_regress1'

    log_dir = 'downstream_result/ADMETlab_data/CycPeptMPDB_Peptide'
    warmup_steps = 0.1#0.1
    log_project = 'cype'
    imbsampler = False # 处理样本不均问题

@ex.named_config
def task_finetune_MoleculeNet_regress_FGFR():
    data_root = './data/downstreamDataset/ADMETlab_data/FGFR_regress'
    #data_root = './data/downstreamDataset/ADMETlab_data/FDA'
    #data_root = './data/downstreamDataset/ADMETlab_data/CycPeptMPDB_Peptide3'
    checkpoint_save_path = './cpkt_path/ADMETlab_data1/FGFR_regress'

    log_dir = 'downstream_result/ADMETlab_data/CycPeptMPDB_Peptide'
    warmup_steps = 0.1#0.1
    log_project = 'cype'
    imbsampler = False # 处理样本不均问题

@ex.named_config
def task_finetune_MoleculeNet_regress_KRAS():
    data_root = './data/downstreamDataset/ADMETlab_data/kras'
    #data_root = './data/downstreamDataset/ADMETlab_data/FDA'
    #data_root = './data/downstreamDataset/ADMETlab_data/CycPeptMPDB_Peptide3'
    checkpoint_save_path = './cpkt_path/ADMETlab_data1/kras'

    log_dir = 'downstream_result/ADMETlab_data/CycPeptMPDB_Peptide'
    warmup_steps = 0.1#0.1
    log_project = 'cype'
    imbsampler = False # 处理样本不均问题

@ex.named_config
def task_finetune_MoleculeNet_regress_Lipo():
    data_root = './data/downstreamDataset/MoleculeNet/Lipo'
    checkpoint_save_path = "./cpkt_path/Molecule/Lipo"
    log_dir = 'downstream_result/MoleculeNet/Lipo'
    batch_size = 128
    max_steps = 1000
    learning_rate = 1e-6



@ex.named_config
def task_oscr_finturn():
    exp_name = "oscr_finturn"
    datasets = ['ocsr_finturn']     # 直接映射到分类数据加载模块
    loss_names = _loss_names({"ocsr_finturn": 1,"mlm": 1})
    data_root = '../data/ocsr_finturn'
    log_dir = 'downstream_result/ocsr_finturn'
    checkpoint_save_path = "./cpkt_path/ocsr"
    is_pretrain = False
    load_path =  ''
    max_steps = 3000
    batch_size = 2048
    per_gpu_batchsize = 8

@ex.named_config
def task_ocsr():
    exp_name = "task_ocsr"
    datasets = ['task_ocsr']     # 直接映射到分类数据加载模块
    loss_names = _loss_names({"ocsr": 1})
    data_root = './data/ocsr/standard/regconResult'
    log_dir = 'downstream_result/ocsr'
    is_pretrain = False
    load_path = ''
    per_gpu_batchsize = 16
    seed = 0
    test_only = True

@ex.named_config
def task_inter():
    exp_name = "task_inter"
    datasets = ['task_inter']     # 直接映射到分类数据加载模块
    loss_names = _loss_names({"inter": 1})
    data_root = './inter/images2000_2' # './data/downstreamDataset/ADMETlab_data/CYP3A4_sub'
    log_dir = 'downstream_result/inter/images2000_2'
    is_pretrain = False
    load_path =  ''
    per_gpu_batchsize = 16
    seed = 0
    test_only = True


#################################  EGFR ####################################
@ex.named_config
def task_finetune_EGFR_classify():
    exp_name = "task_egfr_feature"
    datasets = ['MoleculeNet_classify']     # 直接映射到分类数据加载模块
    loss_names = _loss_names({"MoleculeNet_classify": 1})

    # batch_size = [32,64,128]
    # max_steps = [1000,1500,2000,2500] # 2500
    # learning_rate = [5e-6,1e-5,5e-5]

    warmup_steps = 0.1
    is_pretrain = False       # 是否是预训练阶段
    load_path = ''
    # load_path = ""
    per_gpu_batchsize = 16
    seed = 0
    test_only = False
    mode = 'max'   # 监测验证集最大值时的checkpoint

@ex.named_config
def task_finetune_EGFR_classify_feature():
    data_root = '../data/EGFR/Feature'
    checkpoint_save_path = './cpkt_path/EGFR/Feature'
    log_dir = 'downstream_result/EGFR/Feature'
    log_project = 'EGFR_Feature'
    imbsampler = True

@ex.named_config
def task_finetune_EGFR_classify_past():
    data_root = '../data/EGFR/Past'
    checkpoint_save_path = './cpkt_path/EGFR/Past'
    log_dir = 'downstream_result/EGFR/Past'
    log_project = 'EGFR_Past'

@ex.named_config
def task_finetune_EGFR_classify_feature_imageMol():
    data_root = '../data/EGFR/Feature'
    checkpoint_save_path = './cpkt_path/EGFR/Feature/imageMol'
    log_dir = 'downstream_result/EGFR/Feature_imageMol'
    log_project = 'EGFR_Feature_imageMol'

@ex.named_config
def task_finetune_EGFR_classify_past_imageMol():
    data_root = '../data/EGFR/Past'
    checkpoint_save_path = './cpkt_path/EGFR/Past/imageMol'
    log_dir = 'downstream_result/EGFR/Past_imageMol'
    log_project = 'EGFR_Past_imageMol'


@ex.named_config
def task_finetune_EGFR_classify_feature_chemberta():
    data_root = '../data/EGFR/Feature'
    checkpoint_save_path = './cpkt_path/EGFR/Feature/chemberta'
    log_dir = 'downstream_result/EGFR/Feature_chemberta'
    log_project = 'EGFR_Feature_chemberta'

@ex.named_config
def task_finetune_EGFR_classify_past_chemberta():
    data_root = '../data/EGFR/Past'
    checkpoint_save_path = './cpkt_path/EGFR/Past/chemberta'
    log_dir = 'downstream_result/EGFR/Past_chemberta'
    log_project = 'EGFR_Past_chemberta'

