from .augment_transform import load_norm_transform
from .base_dataset import BaseDataset

class Pretrain_dataset(BaseDataset):
    def __init__(self, *args, image_size,max_smiles_len, split="",is_pretrain=True, **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        self.smiles_column_name = 'Smiles'
#指定 SMILES 列的名称。
        self.labelled = (split == 'train')
#判断是否为训练集，决定是否有标签。

        self.transform = load_norm_transform(image_size)
        self.max_smiles_len = max_smiles_len
        self.is_pretrain = is_pretrain

        #根据 split 值，设置数据集名称。这里都设置为 "test"，实际使用时可能需要根据需要修改。
        if split == "train":
            names = ["test"]
        elif split == "val":
            names = ["test"]
        elif split == "test":
            names = [
                # "smiles_image_pairs_30w7wtest",
                'test'
            ]

        super().__init__(*args, **kwargs, transforms=self.transform, names=names, image_size=image_size, smiles_column_name = self.smiles_column_name,split = self.split,is_pretrain=self.is_pretrain)

    def __getitem__(self, index):
        return self.getItem(index)
#调用 BaseDataset 的初始化方法，传递必要的参数，包括转换方法、数据集名称、图像大小、SMILES 列名称、数据集分割类型和预训练标志。return image,image_false,smiles,smiles_false,k_100,k_1000,k_10000
