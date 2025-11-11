import math
from collections.abc import Callable as _Callable
from typing import override as _override

import torch
from lightning import LightningDataModule
from lightning.fabric.utilities import suggested_max_num_workers as max_num_workers
from numpy import ndarray
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, VisionDataset
from torchvision.transforms import v2


class SZT(VisionDataset):
    classes: list[int] = [0, 1]  # binary classification
    training_file: str = "/SZT.pt"
    transform: _Callable[..., torch.Tensor] | None = (
        None  # initialise variable for typechecker
    )

    def __init__(
        self,
        root: str,
        train: bool,  # only for compatibility
        transform: _Callable[..., torch.Tensor] | None = None,
        download: bool = False,  # only for compatibility
    ):
        super().__init__(root, transform=transform)  # pyright: ignore[reportUnknownMemberType]
        self.data: torch.Tensor
        self.data, self.targets = torch.load(root + self.training_file)
        self.targets: torch.Tensor = self.targets.long()

    @_override
    def __getitem__(self, index: int):
        # treat the data as images
        img = self.data[index].unsqueeze(dim=0)
        target = self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    @_override
    def __len__(self):
        return self.data.size(0)


# has to be after definition of SZT
DATASET_TYPE = type[CIFAR10 | FashionMNIST | MNIST | SZT]


class DataModule(LightningDataModule):
    # initialise variables for typechecker
    input_size: torch.Size = torch.Size([])
    num_classes: int = 0

    train_class_counts: torch.Tensor = torch.empty([])
    train_labels: torch.Tensor = torch.empty([])
    train_data: torch.Tensor = torch.empty([])
    train: TensorDataset = TensorDataset()

    val_class_counts: torch.Tensor = torch.empty([])
    val_labels: torch.Tensor = torch.empty([])
    val_data: torch.Tensor = torch.empty([])
    val: TensorDataset = TensorDataset()

    def __init__(
        self,
        subset: int,
        dataset: DATASET_TYPE,
        data_dir: str,
        batch_size: int,
        num_devices: int,
    ):
        super().__init__()

        self.subset: int = subset
        self.dataset: DATASET_TYPE = dataset
        self.data_dir: str = data_dir
        self.batch_size: int = batch_size
        self.num_devices: int = num_devices
        self.transform: _Callable[..., torch.Tensor] = v2.Compose(
            [
                v2.ToImage(),  # convert to TVTensor Image
                v2.ToDtype(torch.float, scale=True),  # uint8 {0,…,255} to float32 [0,1]
            ]
        )
        self.num_decoders: dict[str, int] = {"train": 1, "val": 1}

    @_override
    def prepare_data(self):
        for train in (True, False):
            _ = self.dataset(self.data_dir, train=train, download=True)

    def _preprocess(self, raw_data: torch.Tensor | ndarray):
        """Reshape data to be at least 4D with dimensions (N, C, H, W)"""
        data = torch.as_tensor(raw_data)
        size_4d = data.size() + (1,) * (4 - data.dim())
        return (
            data.reshape(size_4d)
            .movedim(3, 1)
            .to(torch.float32, memory_format=torch.channels_last)
        )

    def _get_stratification_mask(
        self, labels: torch.Tensor, desired_size: int, subset: int
    ):
        """
        Split `labels` into balanced subsets of size `desired_size` and
        return Boolean mask of the samples in the `subset`th subset.
        Undefined for large `subset` if the dataset is imbalanced.
        """
        assert 0 <= subset < labels.size(0) // desired_size, (
            "Subset index out of range for the given desired size"
        )
        assert desired_size % self.num_classes == 0, (
            "Desired size must be divisible by the number of classes"
        )
        desired_class_size = desired_size // self.num_classes

        # enumerate samples of the same class
        cum_counts = one_hot(labels, self.num_classes).cumsum(0)
        indices = cum_counts[torch.arange(labels.size(0)), labels] - 1

        # shift indices and compute mask
        indices -= desired_class_size * subset
        mask = (0 <= indices) & (indices < desired_class_size)
        assert (size := mask.sum()) == desired_size, (
            f"Size of mask {size} ≠ desired size {desired_size}"
        )

        return mask

    @_override
    def setup(self, stage: str):
        """
        Setup the train and validation datasets.
        Before training, compute the input size, number of classes,
        and class distribution/counts for both datasets.
        Instead of the full training set,
        a stratified subset is used with the same size as the validation set.
        """
        match stage:
            case "fit":
                # use test set for validation (okay since no model selection is done)
                train_ds, val_ds = [
                    self.dataset(self.data_dir, train=train, transform=self.transform)
                    for train in (True, False)
                ]

                # validation set metrics
                self.input_size = val_ds[0][0].size()
                self.num_classes = len(val_ds.classes)
                self.val_labels = torch.as_tensor(val_ds.targets)
                self.val_class_counts = self.val_labels.bincount()

                # compute stratification mask
                labels = torch.as_tensor(train_ds.targets)
                mask = self._get_stratification_mask(
                    labels, self.val_labels.size(0), self.subset
                )

                # train set metrics
                train_input_size = train_ds[0][0].size()
                train_num_classes = len(train_ds.classes)
                self.train_labels = labels[mask]
                self.train_class_counts = self.train_labels.bincount()

                # sanity checks
                assert self.input_size == train_input_size, (
                    "Input size of train set does not match that of val set"
                )
                assert self.num_classes == train_num_classes, (
                    "Number of classes in train set does not match that of validation set"
                )
                for name, num_classes, class_counts in zip(
                    ("Train", "Validation"),
                    (train_num_classes, self.num_classes),
                    (self.train_class_counts, self.val_class_counts),
                ):
                    assert num_classes == (count := class_counts.size(0)), (
                        f"{name} dataset supposedly has {num_classes} classes, "
                        f"but only {count} different labels found."
                    )

                # define datasets
                self.train_data = self._preprocess(train_ds.data[mask])
                self.val_data = self._preprocess(val_ds.data)
                self.train = TensorDataset(self.train_data, self.train_labels)
                self.val = TensorDataset(self.val_data, self.val_labels)
            case _:
                pass

    def _dataloader(self, dataset: torch.utils.data.TensorDataset, shuffle: bool):
        return DataLoader(
            dataset,
            batch_size=self.batch_size // self.num_devices,
            shuffle=shuffle,
            pin_memory=torch.cuda.is_available(),
            num_workers=max_num_workers(self.num_devices),
            persistent_workers=True,  # Keep workers alive between epochs
        )

    @_override
    def train_dataloader(self):
        return self._dataloader(self.train, shuffle=True)

    @_override
    def val_dataloader(self):
        return self._dataloader(self.val, shuffle=True)


class DIBData(DataModule):
    def __init__(self, dm: DataModule):
        super().__init__(
            dm.subset, dm.dataset, dm.data_dir, dm.batch_size, dm.num_devices
        )
        self.labels: dict[str, torch.Tensor] = {"train": dm.train_labels, "val": dm.val_labels}
        self.class_counts: dict[str, torch.Tensor] = {"train": dm.train_class_counts, "val": dm.val_class_counts}
        self.data: dict[str, torch.Tensor] = {"train": dm.train_data, "val": dm.val_data}
        self.num_classes: int = dm.num_classes
        self.num_decoders: dict[str, int] = {
            # ⌈log_C(max_c n_c)⌉ new labels per sample
            dataset: math.ceil(math.log(class_counts.max(), self.num_classes))
            for dataset, class_counts in self.class_counts.items()
        }

    def _base_expand(self, labels: torch.Tensor, num_decoders: int):
        r"""
        Generate relabeling `N` of the data where `N[i,:]` are the new labels of the `i`th sample.
        Each sample obtains ⌈log_C(max_c n_c)⌉ new labels,
        where n_1,…n_C are the class_counts.
        Based on Algorithm 1 of "Learning Optimal Representations with
        the Decodable Information Bottleneck".
        """
        assert labels.dim() == 1, "labels must be 1D"
        assert torch.max(labels) < self.num_classes, "labels ⊈ {0,…,num_classes-1}"

        # enumerate samples of the same class
        cum_counts = one_hot(labels, self.num_classes).cumsum(0)
        indices = cum_counts[torch.arange(labels.size(0)), labels] - 1

        # base |Y| representation of indices padded with 0s to num_decoders digits
        divisors = torch.tensor(
            [self.num_classes**i for i in range(num_decoders - 1, -1, -1)]
        )
        temp = indices.unsqueeze(1).expand(-1, num_decoders)
        return (temp // divisors) % self.num_classes

    @_override
    def setup(self, stage: str):
        match stage:
            case "fit":
                # create DIB datasets
                train_dib_labels, val_dib_labels = (
                    self._base_expand(self.labels[dataset], self.num_decoders[dataset])
                    for dataset in ("train", "val")
                )
                self.train: TensorDataset = TensorDataset(self.data["train"], train_dib_labels)
                self.val: TensorDataset = TensorDataset(self.data["val"], val_dib_labels)
            case _:
                pass
