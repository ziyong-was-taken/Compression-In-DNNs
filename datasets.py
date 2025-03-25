from lightning import LightningDataModule
import torch
import torchvision.datasets as torchdata
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2

from utils import new_labels


class SZT(VisionDataset):
    classes = [0, 1]  # binary classification
    training_file = "data/SZT.pt"

    def __init__(
        self,
        root,
        train: bool,  # currently useless
        transform=None,
        download: bool = False,  # only for compatibility
    ):
        super().__init__(root, transform=transform)
        self.data: torch.Tensor
        self.data, self.targets = torch.load(self.training_file)
        self.targets: torch.Tensor = self.targets.long()

    def __getitem__(self, index):
        # treat the data as images
        img, target = self.data[index].unsqueeze(dim=0), self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return self.data.size(0)


# has to be after definition of SZT
DATASET_TYPE = type[torchdata.MNIST | torchdata.FashionMNIST | torchdata.CIFAR10 | SZT]


class RelabeledDataset(VisionDataset):
    """Dataset wrapper that relabels the data with `new_labels`"""

    def __init__(self, dataset: VisionDataset, new_labels: torch.Tensor):
        self.dataset = dataset
        self.new_labels = new_labels

    def __getitem__(self, index):
        data, _ = self.dataset[index]
        return data, self.new_labels[index]

    def __len__(self):
        return len(self.dataset)


class DataModule(LightningDataModule):
    def __init__(self, dataset: DATASET_TYPE, data_dir: str):
        super().__init__()

        self.dataset = dataset
        self.data_dir = data_dir
        self.batch_size: int = 1  # dummy value
        self.transform = v2.Compose(
            [  # TODO: image augmentation and/or normalisation
                # v2.Resize(224),  # ResNet18 and ConvNext-Tiny expect 224x224 images
                v2.ToImage(),  # convert to TVTensor Image
                v2.ToDtype(
                    torch.float32, scale=True
                ),  # uint8 {0,…,255} to float32 [0,1]
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
                labels = torch.as_tensor(base_ds.targets)
                self.base_expansion = new_labels(labels, self.num_classes)
                self.train = RelabeledDataset(
                    dataset=base_ds,
                    # one-hot labels compatible with both MSE loss and cross-entropy loss
                    new_labels=one_hot(labels, self.num_classes).float(),
                )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=torch.get_num_threads(),
            persistent_workers=True,  # Keep workers alive between epochs
        )


class DIBData(DataModule):
    def __init__(self, datamodule: DataModule):
        super().__init__(datamodule.dataset, datamodule.data_dir)
        self.base_expansion = datamodule.base_expansion

    def setup(self, stage):
        """Create the train dataset with new labels."""
        match stage:
            case "fit":
                base_ds = self.dataset(
                    self.data_dir, train=True, transform=self.transform
                )
                self.train = RelabeledDataset(
                    dataset=base_ds, new_labels=self.base_expansion
                )
