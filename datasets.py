import math

import torch
from lightning import LightningDataModule
from lightning.fabric.utilities import suggested_max_num_workers as max_num_workers
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, VisionDataset
from torchvision.transforms import v2


class SZT(VisionDataset):
    classes = [0, 1]  # binary classification
    training_file = "/SZT.pt"

    def __init__(
        self,
        root: str,
        train: bool,  # currently useless
        transform=None,
        download: bool = False,  # only for compatibility
    ):
        super().__init__(root, transform=transform)
        self.data: torch.Tensor
        self.data, self.targets = torch.load(root + self.training_file)
        self.targets: torch.Tensor = self.targets.long()

    def __getitem__(self, index):
        # treat the data as images
        img = self.data[index].unsqueeze(dim=0)
        target = self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return self.data.size(0)


# has to be after definition of SZT
DATASET_TYPE = type[CIFAR10 | FashionMNIST | MNIST | SZT]
BASE_DATASET_TYPE = CIFAR10 | FashionMNIST | MNIST | SZT


class DataModule(LightningDataModule):
    def __init__(
        self, dataset: DATASET_TYPE, data_dir: str, batch_size: int, num_devices: int
    ):
        super().__init__()

        self.dataset = dataset
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_devices = num_devices
        self.transform = v2.Compose(
            [  # TODO: image augmentation and/or normalisation
                v2.ToImage(),  # convert to TVTensor Image
                v2.ToDtype(torch.float, scale=True),  # uint8 {0,…,255} to float32 [0,1]
            ]
        )

    def prepare_data(self):
        for train in (True, False):
            self.dataset(self.data_dir, train=train, download=True)

    def _preprocess(self, raw_data):
        """Reshape data to be at least 4D."""
        data = torch.as_tensor(raw_data)
        size_4d = data.size() + (1,) * (4 - data.dim())
        return data.reshape(size_4d).movedim(3, 1).float()

    def setup(self, stage):
        """
        Setup the train and validation datasets.
        Before training, compute the input size, number of classes,
        and class distribution/counts for both datasets.
        """
        match stage:
            case "fit":
                # use test set for validation (okay since no model selection is done)
                train_ds, val_ds = [
                    self.dataset(self.data_dir, train=train, transform=self.transform)
                    for train in (True, False)
                ]

                # train set metrics
                self.input_size = train_ds[0][0].size()
                self.num_classes = len(train_ds.classes)
                self.train_labels = torch.as_tensor(train_ds.targets)
                self.train_class_counts = self.train_labels.bincount()

                # val set metrics
                val_input_size = val_ds[0][0].size()
                val_num_classes = len(val_ds.classes)
                self.val_labels = torch.as_tensor(val_ds.targets)
                self.val_class_counts = self.val_labels.bincount()

                # sanity checks
                assert self.input_size == val_input_size, (
                    "Input size of train set does not match that of val set"
                )
                assert self.num_classes == val_num_classes, (
                    "Number of classes in train set does not match that of validation set"
                )
                for name, num_classes, class_counts in zip(
                    ("Train", "Validation"),
                    (self.num_classes, val_num_classes),
                    (self.train_class_counts, self.val_class_counts),
                ):
                    assert num_classes == (count := class_counts.size(0)), (
                        f"{name} dataset supposedly has {num_classes} classes, "
                        f"but only {count} different labels found."
                    )

                # define datasets
                self.train = TensorDataset(
                    self._preprocess(train_ds.data), self.train_labels
                )
                self.val = TensorDataset(self._preprocess(val_ds.data), self.val_labels)

    def _dataloader(self, dataset, shuffle: bool):
        return DataLoader(
            dataset,
            batch_size=self.batch_size // self.num_devices,
            shuffle=shuffle,
            pin_memory=torch.cuda.is_available(),
            num_workers=max_num_workers(self.num_devices),
            persistent_workers=True,  # Keep workers alive between epochs
        )

    def train_dataloader(self):
        return self._dataloader(self.train, shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.val, shuffle=False)


class DIBData(DataModule):
    def __init__(
        self, dataset: DATASET_TYPE, data_dir: str, batch_size: int, num_devices: int
    ):
        super().__init__(dataset, data_dir, batch_size, num_devices)

    def _base_expand(
        self, labels: torch.Tensor, class_counts: torch.Tensor, num_classes: int
    ):
        r"""
        Generate relabeling `N` of the data where `N[i,:]` are the new labels of the `i`th sample.
        Each sample obtains ⌈log_{|Y|}(max{|X_y| : y ∈ Y})⌉ new labels,
        where {|X_y| : y ∈ Y} = {|{l ∈ `labels` : l = y}| : y ∈ Y} = `class_counts`
        and Y = {0,…,`num_classes`-1}.
        Based on Algorithm 1 of "Learning Optimal Representations with
        the Decodable Information Bottleneck".
        """
        assert labels.dim() == 1, "labels must be 1D"
        assert torch.max(labels) < num_classes, "labels ⊈ {0,…,num_classes-1}"

        # compute ⌈log_{|Y|}(max{|X_y| : y ∈ Y})⌉
        num_digits = math.ceil(math.log(class_counts.max(), num_classes))

        # enumerate samples of the same class
        cum_counts = one_hot(labels, num_classes).cumsum(0)
        idcs = cum_counts[torch.arange(labels.size(0)), labels] - 1

        # base |Y| representation of indices padded with 0s to num_digits digits
        divisors = torch.tensor([num_classes**i for i in range(num_digits - 1, -1, -1)])
        temp = idcs.unsqueeze(1).expand(-1, num_digits)
        return (temp // divisors) % num_classes

    def setup(self, stage):
        match stage:
            case "fit":
                # use test set for validation (okay since no model selection is done)
                train_ds, val_ds = [
                    self.dataset(self.data_dir, train=train, transform=self.transform)
                    for train in (True, False)
                ]

                # create DIB datasets
                train_dib_labels, val_dib_labels = (
                    self._base_expand(labels, class_counts, self.num_classes)
                    for labels, class_counts in zip(
                        (self.train_labels, self.val_labels),
                        (self.train_class_counts, self.val_class_counts),
                    )
                )
                self.train = TensorDataset(
                    self._preprocess(train_ds.data), train_dib_labels
                )
                self.val = TensorDataset(self._preprocess(val_ds.data), val_dib_labels)
