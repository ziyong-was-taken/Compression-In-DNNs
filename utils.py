import argparse
import math

import torch


# defaults for command line arguments
EPOCHS = 1000
BATCH_SIZE = 1024
DIB_EPOCHS = 200


def get_args():
    """
    Parse command line arguments:

    -m, --model: model to use
    -w, --widths: widths of hidden layers of MLP
    -nl, --nonlinearity: nonlinearity used in hidden layers of MLP
    -d, --dataset: dataset to use
    -opt, --optimiser: optimiser to use
    --loss: loss function to use
    --epochs: number of epochs to train the model
    -bs, --batch-size: batch size
    --dib-epochs: number of epochs to train the DIB network
    --dib-batch-size: batch size for the DIB network
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
        default=[10, 7, 5, 4, 3],  # same model as Schwartz-Ziv & Tishby (2017)
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
        "-d",
        "--dataset",
        default="SZT",
        choices=["MNIST", "CIFAR10", "FashionMNIST", "SZT"],
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
    parser.add_argument("--epochs", default=EPOCHS, type=int)
    parser.add_argument("-bs", "--batch-size", default=BATCH_SIZE, type=int)
    parser.add_argument("--dib-epochs", default=DIB_EPOCHS, type=int)
    parser.add_argument("--dib-batch-size", default=BATCH_SIZE, type=int)
    return parser.parse_args()


def new_labels(labels: torch.Tensor, num_classes: int):
    r"""
    Generate random relabeling `N` of the data where `N[i,:]` are the new labels of the `i`th sample.
    Each sample obtains ⌊log_{`num_classes`}(max{|X_y| : y ∈ Y} - 1)⌋ + 1 new labels,
    where X_y = {x ∈ X : x.label = y}$ and
    Y = {0,…,`num_classes`-1} is the set of all labels.
    Based on Algorithm 1 of
    "Learning Optimal Representations with the Decodable Information Bottleneck".
    """
    assert labels.dim() == 1, "labels must be 1D"
    assert torch.max(labels) < num_classes, "labels ⊈ {0,…,num_classes}"

    # compute ⌊log_{|Y|}(max{|X_y| : y ∈ Y} - 1)⌋ + 1 = ⌈log_{|Y|}(max{|X_y| : y ∈ Y})⌉
    num_digits = math.ceil(math.log(labels.bincount().max(), num_classes))

    # enumerate samples of the same class
    idcs = torch.zeros_like(labels)
    for y in range(num_classes):
        mask = labels == y
        if (count := mask.sum()) > 0:
            idcs[mask] = torch.arange(0, count)

    # base |Y| representation of indices padded with 0s to num_digits digits
    divisors = torch.tensor([num_classes**i for i in range(num_digits - 1, -1, -1)])
    temp = idcs.unsqueeze(1).expand(-1, num_digits)
    return (temp // divisors) % num_classes
