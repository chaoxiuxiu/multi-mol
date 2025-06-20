from argparse import Namespace
import random
from typing import Callable, List, Union

import numpy as np
from torch.utils.data.dataset import Dataset
from rdkit import Chem

from .scaler import StandardScaler
from chemprop.features import get_features_generator
import pdb



class MoleculeDatapoint:
    """A MoleculeDatapoint contains a single molecule and its associated features and targets."""

    def __init__(self,
                 line: List[str],
                 args: Namespace = None,
                 features: np.ndarray = None,#包含附加特征的 numpy 数组
                 use_compound_names: bool = False):
        """
        Initializes a MoleculeDatapoint, which contains a single molecule.

        :param line: A list of strings generated by separating a line in a data CSV file by comma.
        :param args: Arguments.
        :param features: A numpy array containing additional features (ex. Morgan fingerprint).
        :param use_compound_names: Whether the data CSV includes the compound name on each line.
        """
        if args is not None:
            self.features_generator = args.features_generator
            self.args = args
        else:
            self.features_generator = self.args = None

        if features is not None and self.features_generator is not None:#果同时提供了 features 和 features_generator，则抛出异常。因为两者不能同时存在。
            raise ValueError('Currently cannot provide both loaded features and a features generator.')

        self.features = features

        if use_compound_names:#如果 use_compound_names 为 True，则从 line 中提取化合物名称并存储在实例变量中，同时将 line 的其余部分更新为不包含化合物名称。
            self.compound_name = line[0]  # str
            line = line[1:]
        else:
            self.compound_name = None

        self.smiles = line[0]  # str从 line 中提取 SMILES 字符串并存储在实例变量 self.smiles 中。
        self.mol = Chem.MolFromSmiles(self.smiles)#使用 Chem.MolFromSmiles 函数将 SMILES 字符串转换为 RDKit 分子对象，并存储在 self.mol 中。
        

        # Generate additional features if given a generator如果存在 features_generator，则生成附加特征。
        if self.features_generator is not None:
            self.features = []

            for fg in self.features_generator:#遍历 features_generator 中的每个生成器 fg，使用 get_features_generator 函数获取实际的特征生成器函数。
                features_generator = get_features_generator(fg)
                if self.mol is not None and self.mol.GetNumHeavyAtoms() > 0:#如果 self.mol 不为 None 且含有重原子，则使用特征生成器生成特征并扩展到 self.features 列表中。
                    self.features.extend(features_generator(self.mol))

            self.features = np.array(self.features)#将 self.features 转换为 numpy 数组。

        # Fix nans in features
        if self.features is not None:#    如果 self.features 不为 None，则处理特征中的 NaN 值。将 NaN 值替换为 replace_token（这里是 0）。
            replace_token = 0
            self.features = np.where(np.isnan(self.features), replace_token, self.features)

        # Create targets
        self.targets = [float(x) if x != '' else None for x in line[1:]]#    从 line 中的剩余部分（除了第一个元素）创建目标值列表。将非空字符串转换为浮点数，空字符串转换为 None。

    def set_features(self, features: np.ndarray):
        """
        Sets the features of the molecule.

        :param features: A 1-D numpy array of features for the molecule.
        """
        self.features = features

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.

        :return: The number of tasks.
        """
        return len(self.targets)

    def set_targets(self, targets: List[float]):
        """
        Sets the targets of a molecule.

        :param targets: A list of floats containing the targets.
        """
        self.targets = targets


class MoleculeDataset(Dataset):
    """A MoleculeDataset contains a list of molecules and their associated features and targets."""

    def __init__(self, data: List[MoleculeDatapoint]):
        """
        Initializes a MoleculeDataset, which contains a list of MoleculeDatapoints (i.e. a list of molecules).

        :param data: A list of MoleculeDatapoints.
        """
        self.data = data
        self.args = self.data[0].args if len(self.data) > 0 else None
        self.scaler = None

    def compound_names(self) -> List[str]:
        """
        Returns the compound names associated with the molecule (if they exist).

        :return: A list of compound names or None if the dataset does not contain compound names.
        """
        if len(self.data) == 0 or self.data[0].compound_name is None:
            return None

        return [d.compound_name for d in self.data]

    def smiles(self) -> List[str]:
        """
        Returns the smiles strings associated with the molecules.

        :return: A list of smiles strings.
        """
        return [d.smiles for d in self.data]
    
    def mols(self) -> List[Chem.Mol]:
        """
        Returns the RDKit molecules associated with the molecules.

        :return: A list of RDKit Mols.
        """
        return [d.mol for d in self.data]

    def features(self) -> List[np.ndarray]:
        """
        Returns the features associated with each molecule (if they exist).

        :return: A list of 1D numpy arrays containing the features for each molecule or None if there are no features.
        """
        if len(self.data) == 0 or self.data[0].features is None:
            return None

        return [d.features for d in self.data]

    def targets(self) -> List[List[float]]:
        """
        Returns the targets associated with each molecule.

        :return: A list of lists of floats containing the targets.
        """
        return [d.targets for d in self.data]

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.

        :return: The number of tasks.
        """
        return self.data[0].num_tasks() if len(self.data) > 0 else None

    def features_size(self) -> int:
        """
        Returns the size of the features array associated with each molecule.

        :return: The size of the features.
        """
        return len(self.data[0].features) if len(self.data) > 0 and self.data[0].features is not None else None

    def shuffle(self, seed: int = None):
        """
        Shuffles the dataset.

        :param seed: Optional random seed.
        """
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.data)
    
    def normalize_features(self, scaler: StandardScaler = None, replace_nan_token: int = 0) -> StandardScaler:
        """
        Normalizes the features of the dataset using a StandardScaler (subtract mean, divide by standard deviation).

        If a scaler is provided, uses that scaler to perform the normalization. Otherwise fits a scaler to the
        features in the dataset and then performs the normalization.

        :param scaler: A fitted StandardScaler. Used if provided. Otherwise a StandardScaler is fit on
        this dataset and is then used.
        :param replace_nan_token: What to replace nans with.
        :return: A fitted StandardScaler. If a scaler is provided, this is the same scaler. Otherwise, this is
        a scaler fit on this dataset.
        """
        if len(self.data) == 0 or self.data[0].features is None:
            return None

        if scaler is not None:
            self.scaler = scaler

        elif self.scaler is None:
            features = np.vstack([d.features for d in self.data])
            self.scaler = StandardScaler(replace_nan_token=replace_nan_token)
            self.scaler.fit(features)

        for d in self.data:
            d.set_features(self.scaler.transform(d.features.reshape(1, -1))[0])

        return self.scaler
    
    def set_targets(self, targets: List[List[float]]):
        """
        Sets the targets for each molecule in the dataset. Assumes the targets are aligned with the datapoints.

        :param targets: A list of lists of floats containing targets for each molecule. This must be the
        same length as the underlying dataset.
        """
        assert len(self.data) == len(targets)
        for i in range(len(self.data)):
            self.data[i].set_targets(targets[i])

    def sort(self, key: Callable):
        """
        Sorts the dataset using the provided key.

        :param key: A function on a MoleculeDatapoint to determine the sorting order.
        """
        self.data.sort(key=key)

    def __len__(self) -> int:
        """
        Returns the length of the dataset (i.e. the number of molecules).

        :return: The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, item) -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:
        """
        Gets one or more MoleculeDatapoints via an index or slice.

        :param item: An index (int) or a slice object.
        :return: A MoleculeDatapoint if an int is provided or a list of MoleculeDatapoints if a slice is provided.
        """
        return self.data[item]
