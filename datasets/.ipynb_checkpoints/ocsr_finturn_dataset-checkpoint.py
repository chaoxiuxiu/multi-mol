from .augment_transform import load_norm_transform
from .base_dataset import BaseDataset

class ocsr_finturn_dataset(BaseDataset):
    def __init__(self, *args, image_size,max_smiles_len, split="train",is_pretrain=False, **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        self.smiles_column_name = 'Smiles'
        self.is_pretrain = is_pretrain

        self.transform = load_norm_transform(image_size)
        self.max_smiles_len = max_smiles_len

        if split == "train":
            names = ["train"]
        elif split == "val":
            names = ["test"]
        elif split == "test":
            names = ["test"]

        super().__init__(*args, **kwargs, transforms=self.transform, names=names, image_size=image_size, smiles_column_name = self.smiles_column_name,split = self.split,is_pretrain=self.is_pretrain,MoleculeNet=False,ocsr=True)

    def __getitem__(self, index):
        return self.ocsr_getItem(index)