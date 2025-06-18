from .augment_transform import load_norm_transform
from .base_dataset import BaseDataset

class inter_dataset(BaseDataset):
    def __init__(self, data_dir, image_size,max_smiles_len, split="predict",is_pretrain=False, **kwargs):
        assert split in ["train", "val", "test", "predict",]
        self.split = split
        self.smiles_column_name = 'Smiles'
        self.is_pretrain = is_pretrain
        self.data_dir = data_dir
        self.transform = load_norm_transform(image_size)
        self.max_smiles_len = max_smiles_len

        if split == "train":
            names = ["train",'dev','test']
        elif split == "val":
            names = ["dev"]
        elif split == "test":
            names = ["test"]
        elif split == "predict":
            names = ["predict"]

        super().__init__(self.data_dir,transforms=self.transform, names=names, image_size=image_size,  smiles_column_name = self.smiles_column_name,label_name='labels',split = self.split,is_pretrain=self.is_pretrain,MoleculeNet=True,ocsr=False)

    def __getitem__(self, index):
        return self.get_smiles(index),self.get_image(index),self.get_label(index)