import torch
from lightning import LightningDataModule
from lightning.fabric.utilities import suggested_max_num_workers as max_num_workers
from lightning.fabric.utilities.rank_zero import _get_rank
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets as torchdata
from torchvision.datasets import VisionDataset
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
DATASET_TYPE = type[torchdata.CIFAR10 | torchdata.FashionMNIST | torchdata.MNIST | SZT]


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
                # v2.Resize(224),  # ResNet18 and ConvNext-Tiny expect 224x224 images
                v2.ToImage(),  # convert to TVTensor Image
                v2.ToDtype(torch.float, scale=True),  # uint8 {0,…,255} to float32 [0,1]
            ]
        )

    def prepare_data(self):
        for train in (True, False):
            self.dataset(self.data_dir, train=train, download=True)

    def _preprocess(self, raw_data):
        """
        Move data to the current device and reshape it to be at least 4D.
        Not part of `prepare_data` since `prepare_data` is only called on rank 0.
        """
        device = torch.device(
            f"cuda:{_get_rank()}" if torch.cuda.is_available() else "cpu"
        )
        data = torch.as_tensor(raw_data, device=device)
        size_4d = data.size() + (1,) * (4 - data.dim())
        return data.reshape(size_4d).movedim(3, 1).float()

    def setup(self, stage):
        """
        Setup the train and test datasets.
        When training, compute the
        - input size,
        - output size,
        - class distribution/counts, and
        - DIB labels.
        """
        match stage:
            case "fit":
                base_ds = self.dataset(
                    self.data_dir, train=True, transform=self.transform
                )

                # TODO: train-validation split
                self.input_size = base_ds[0][0].size()
                self.num_classes = len(base_ds.classes)
                self.labels = torch.as_tensor(base_ds.targets)
                self.class_counts = self.labels.bincount()
                assert self.num_classes == (count := self.class_counts.size(0)), (
                    f"Dataset supposedly has {self.num_classes} classes, "
                    f"but only {count} different labels found"
                )

                self.train = TensorDataset(
                    self._preprocess(base_ds.data),
                    self.labels,
                    # self.labels[torch.randperm(len(self.labels))],
                )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size // self.num_devices,
            shuffle=True,
            num_workers=max_num_workers(self.num_devices),
            persistent_workers=True,  # Keep workers alive between epochs
        )


class DIBData(DataModule):
    def __init__(self, datamodule: DataModule, dib_labels: torch.Tensor):
        super().__init__(
            datamodule.dataset,
            datamodule.data_dir,
            datamodule.batch_size,
            datamodule.num_devices,
        )
        self.dib_labels = dib_labels

    def setup(self, stage):
        """Create the train dataset with new labels."""
        match stage:
            case "fit":
                base_ds = self.dataset(
                    self.data_dir, train=True, transform=self.transform
                )
                self.train = TensorDataset(
                    self._preprocess(base_ds.data), self.dib_labels
                )
