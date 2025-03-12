import math

import torch
from torch import nn, optim
from torchvision import datasets

# constants for type hinting
DATASET_TYPE = type[datasets.MNIST | datasets.FashionMNIST | datasets.CIFAR10]
NL_TYPE = type[nn.ReLU | nn.Tanh]
LOSS_TYPE = type[nn.CrossEntropyLoss | nn.MSELoss]
OPTIMISER_TYPE = type[optim.AdamW | optim.Adam | optim.SGD]


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
    num_digits = math.ceil(math.log(labels.bincount().max(), num_classes))

    # Algorithm 1 of dib-2020
    idcs = torch.zeros_like(labels)
    for y in range(num_classes):
        mask = labels == y
        idcs[mask] = torch.arange(0, torch.sum(mask).item())
    N = change_base(idcs, num_classes, num_digits)

    return N


def change_base(ints: torch.Tensor, base: int, num_digits: int):
    """
    Convert a 1D tensor of integers `ints` to their base `base` representation
    with the most significant digit on the left,
    padding with 0s to end up with `num_digits` digits.
    Returns a tensor of shape `(len(ints), num_digits)`.
    """
    assert ints.dim() == 1, "ints must be 1D"

    digits = torch.zeros(ints.numel(), num_digits, dtype=torch.long)
    for i in range(num_digits):
        digits[:, num_digits - i - 1] = ints % base
        ints = ints // base
    return digits
