from .MoleculeNetClassify_datamodule import MoleculeNetClassifyDataModule
from .MoleculeNetRegress_datamodule import MoleculeNetRegressDataModule
from .oscrFinturn_datamodule import oscrFinturnDataModule

from .pre_datamodule import pretrainDataModule

_datamodules = {
    "pretrain": pretrainDataModule,
    'MoleculeNet_classify': MoleculeNetClassifyDataModule,
    'MoleculeNet_regress': MoleculeNetRegressDataModule,
    'ocsr_finturn': oscrFinturnDataModule
}