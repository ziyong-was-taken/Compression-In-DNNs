import torch
from lightning import LightningDataModule
from lightning.fabric.utilities import suggested_max_num_workers as max_num_workers
from torch.nn.functional import one_hot
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
DATASET_TYPE = type[torchdata.MNIST | torchdata.FashionMNIST | torchdata.CIFAR10 | SZT]


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

    def setup(self, stage):
        """
        Setup the train and test datasets.
        When training, compute the input and output size as well as the new labels.
        """
        match stage:
            case "fit":
                base_ds = self.dataset(
                    self.data_dir, train=True, transform=self.transform
                )
                self.input_size = base_ds[0][0].size()
                self.num_classes = len(base_ds.classes)
                self.labels = torch.as_tensor(base_ds.targets)
                self.class_counts = self.labels.bincount()
                self.train = TensorDataset(
                    base_ds.data.float(), one_hot(self.labels, self.num_classes).float()
                )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
            num_workers=max_num_workers(self.num_devices),
            persistent_workers=True,  # Keep workers alive between epochs
        )


class DIBData(DataModule):
    def __init__(self, datamodule: DataModule, new_labels: torch.Tensor):
        super().__init__(
            datamodule.dataset,
            datamodule.data_dir,
            datamodule.batch_size,
            datamodule.num_devices,
        )
        self.new_labels = new_labels

    def setup(self, stage):
        """Create the train dataset with new labels."""
        match stage:
            case "fit":
                base_ds = self.dataset(
                    self.data_dir, train=True, transform=self.transform
                )
                self.train = TensorDataset(base_ds.data.float(), self.new_labels)
