import argparse
import math

import torch
from torch import nn, optim
from torchvision import datasets

# constants for type hinting
DATASET_TYPE = type[datasets.MNIST | datasets.FashionMNIST | datasets.CIFAR10]
NL_TYPE = type[nn.ReLU | nn.Tanh]
LOSS_TYPE = type[nn.CrossEntropyLoss | nn.MSELoss]
OPTIMISER_TYPE = type[optim.AdamW | optim.Adam | optim.SGD]


def get_args():
    """
    Parse command line arguments:

    -m, --model: model to use
    -w, --widths: widths of hidden layers of MLP
    -nl, --nonlinearity: nonlinearity used in hidden layers of MLP
    --dataset: dataset to use
    -opt, --optimiser: optimiser to use
    --loss: loss function to use
    --epochs: number of epochs to train the model
    -bs, --batch-size: batch size
    --decoder-epochs: number of epochs to train the decoder
    --decoder-batch-size: batch size for the decoder
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        default="MLP",
        choices=["MLP", "ConvNeXt", "ResNet"],
        help="model to use",
    )
    parser.add_argument(
        "-w",
        "--widths",
        nargs="+",
        default=[12, 10, 7, 5, 4, 3, 2],  # same model as ib-2017
        type=int,
        help="widths of hidden layers of MLP, has no effect on ConvNeXt and ResNet (yet)",
        metavar="WIDTH",
    )
    parser.add_argument(
        "-nl",
        "--nonlinearity",
        default="ReLU",
        help=(
            "nonlinearity used in hidden layers of MLP, "
            "has no effect on ConvNeXt and ResNet (yet), "
            "case-sensitive"
        ),
    )
    parser.add_argument(
        "--dataset",
        default="MNIST",
        choices=["MNIST", "CIFAR10", "FashionMNIST"],
        help="dataset to use, case-sensitive",
    )
    parser.add_argument(
        "-opt",
        "--optimiser",
        default="AdamW",
        choices=["AdamW", "Adam", "SGD"],
        help="optimiser to use, case-sensitive",
    )
    parser.add_argument(
        "--loss",
        default="CrossEntropy",
        choices=["CrossEntropy", "MSE"],
        help="loss function to use, case-sensitive",
    )
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("-bs", "--batch-size", default=64, type=int)
    parser.add_argument("--decoder-epochs", default=5, type=int)
    parser.add_argument("--decoder-batch-size", default=64, type=int)
    return parser.parse_args()


def convert_to_classes(
    dataset_name, nonlinearity_name, loss_name, optimiser_name
) -> tuple[DATASET_TYPE, NL_TYPE, LOSS_TYPE, OPTIMISER_TYPE]:
    """Convert string names to classes"""
    return (
        getattr(datasets, dataset_name),
        getattr(nn, nonlinearity_name),
        getattr(nn, loss_name + "Loss"),
        getattr(optim, optimiser_name),
    )


def new_labels(labels: torch.Tensor, num_classes: int):
    r"""
    Generate random relabeling `N` of the data where `N[i,:]` are the new labels of the `i`th sample.
    Each sample obtains ⌊log_{`num_classes`}(max{|X_y| : y ∈ Y} - 1)⌋ + 1 new labels,
    where X_y = {x ∈ X : x.label = y}$ and Y = {0,…,`num_classes`-1} is the set of all labels.
    """
    assert labels.dim() == 1, "labels must be 1D"
    assert torch.max(labels) < num_classes, "labels ⊈ {0,…,num_classes}"

    # compute ⌊log_{|Y|}(max{|X_y| : y ∈ Y} - 1)⌋ + 1 = ⌈log_{|Y|}(max{|X_y| : y ∈ Y})⌉
    num_digits = math.ceil(math.log(labels.bincount().max().item(), num_classes))

    # Algorithm 1 of dib-2020
    idcs = torch.zeros_like(labels)
    for y in range(num_classes):
        mask = labels == y
        if (count := mask.sum().item()) > 0:
            idcs[mask] = torch.arange(0, count)

    # base |Y| representation of indices padded with 0s to num_digits digits
    divisors = torch.tensor([num_classes**i for i in range(num_digits - 1, -1, -1)])
    temp = idcs.unsqueeze(1).expand(-1, num_digits)
    return (temp // divisors) % num_classes
