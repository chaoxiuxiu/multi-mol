import functools

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torchsampler import ImbalancedDatasetSampler

from datasets.inter_dataset import inter_dataset
from . import _datamodules

# 数据模块，设置并行
class inter_MTDataModule():
    def __init__(self, _config, dist=False):
        datamodule_keys = _config["datasets"]   # ["oscr_pair"]
        assert len(datamodule_keys) > 0
        super().__init__()

        self.dist = dist
        self.data_dir = _config["data_root"]
        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]
        self.image_size = _config["image_size"]
        self.max_smiles_len = _config["max_smiles_len"]

        self.predict_dataset = inter_dataset(self.data_dir,self.image_size,self.max_smiles_len, split="predict",is_pretrain=False,)

    def get_dataloader(self):
        if self.dist:
            self.sampler = DistributedSampler(self.predict_dataset, shuffle=False)
        else:
            self.sampler = None

        loader = DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=False,
        )
        return loader