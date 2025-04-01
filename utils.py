import argparse
import math

import torch
from lightning import Callback, Trainer
from lightning.pytorch.callbacks import EarlyStopping

from datasets import DIBData
from networks import DIBNetwork, MetricNetwork


# defaults for command line arguments
EPOCHS = 1000
BATCH_SIZE = 64
LR = 1e-3
NUM_DEVICES = 1
DIB_EPOCHS = 200


def get_args():
    """
    Parse command line arguments:

    -m, --model: model to use
    -w, --widths: widths of hidden layers of MLP
    -nl, --nonlinearity: nonlinearity used in hidden layers of MLP
    -d, --dataset: dataset to use
    --data-dir: directory containing the dataset
    -opt, --optimiser: optimiser to use
    --loss: loss function to use
    --epochs: number of epochs to train the model
    --num-devices: number of devices used to train the DIB network
    --dib-epochs: number of epochs to train the DIB network
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", default="MLP", choices=["MLP", "ConvNeXt", "ResNet"]
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
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "-opt", "--optimiser", default="AdamW", choices=["AdamW", "Adam", "SGD"]
    )
    parser.add_argument(
        "--loss", default="CrossEntropy", choices=["CrossEntropy", "MSE"]
    )
    parser.add_argument("--epochs", default=EPOCHS, type=int)
    parser.add_argument("--batch-size", default=BATCH_SIZE, type=int)
    parser.add_argument("-lr", "--learning-rate", default=LR, type=float)
    parser.add_argument(
        "--num-devices",
        default=NUM_DEVICES,
        type=int,
        help="number of devices used to train the DIB network",
    )
    parser.add_argument("--dib-epochs", default=DIB_EPOCHS, type=int)
    return parser.parse_args()


def base_expand(labels: torch.Tensor, num_classes: int):
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


class ComputeNC1(Callback):
    """Compute and log NC metrics at the end of each epoch"""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.class_counts = torch.zeros(num_classes)
        # zˡ: size of flattened activations of layer l
        # layer_metrics[l]: ((num_classes + zˡ) × zˡ)
        #     layer_metrics[l][:num_classes]: class sums of activations (num_classes × zˡ)
        #     layer_metrics[l][num_classes:]: gram matrix (zˡ × zˡ)
        self.layer_metrics: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def on_train_batch_end(
        self,
        _trainer,
        network: MetricNetwork,
        _outputs,
        batch: list[torch.Tensor],
        _batch_idx,
    ):
        """Update NC metrics after each batch"""
        _, targets = batch

        # only update class counts on first epoch
        if network.current_epoch == 0:
            self.class_counts = self.class_counts.to(targets)
            self.class_counts += targets.sum(dim=0)

        # update other metrics
        for layer, activations in network.batch_activations.items():
            activations = activations.flatten(start_dim=1)
            if layer not in self.layer_metrics:
                act_size = activations.size(1)
                self.layer_metrics[layer] = torch.zeros(
                    (self.num_classes + act_size, act_size)
                ).to(activations)
            self.layer_metrics[layer][: self.num_classes] += targets.T @ activations
            self.layer_metrics[layer][self.num_classes :] += activations.T @ activations

    @torch.no_grad()
    def on_train_epoch_end(self, _trainer, network: MetricNetwork):
        """Aggregate batched NC metrics and log them"""
        nc: dict[str, torch.Tensor] = {}
        total_count = self.class_counts.sum()
        for layer, joint_metric in self.layer_metrics.items():
            class_sums = joint_metric[: self.num_classes]
            class_means = class_sums / self.class_counts.unsqueeze(dim=1)
            global_mean = class_sums.sum(dim=0) / total_count
            centred_means = class_means - global_mean
            between_cov = (centred_means.T @ centred_means) / self.num_classes
            within_cov = (
                joint_metric[self.num_classes :] / total_count
                - torch.outer(global_mean, global_mean)
                - between_cov
            )
            nc[layer] = torch.linalg.lstsq(
                between_cov, within_cov, rcond=1e-5
            ).solution.trace()
        network.log_dict(nc)

        # reset NC metrics for next epoch (class counts don't change)
        self.layer_metrics.clear()


class ComputeDIB(Callback):
    """Compute and log the DIB metric at the end of each epoch"""

    def __init__(
        self,
        num_decoders: int,
        dib_epochs: int,
        dib_dm: DIBData,
        num_devices: int,
        block_indices: list[int],
    ):
        self.num_decoders = num_decoders
        self.dib_epochs = dib_epochs
        self.dib_dm = dib_dm
        self.num_devices = num_devices
        self.block_indices = block_indices
        self.dib_nets: list[DIBNetwork] = [None for _ in block_indices]

    def on_train_epoch_end(self, _trainer, network: MetricNetwork):
        """Train the DIB network and log the final training loss"""
        for block_idx in self.block_indices:
            # initialise DIB network
            if self.dib_nets[block_idx] is None:
                self.dib_nets[block_idx] = DIBNetwork(
                    *network.get_encoder_decoder(block_idx),
                    self.num_decoders,
                    network.optimiser,
                    network.learning_rate,
                )
                self.dib_nets[block_idx].compile()

            # train DIB network
            dib_trainer = Trainer(
                devices=self.num_devices,
                max_epochs=self.dib_epochs,
                logger=False,  # don't write (but do store) training losses
                default_root_dir="lightning_logs",
                deterministic=True,
                callbacks=[EarlyStopping(monitor="train_loss")],
            )
            dib_trainer.fit(self.dib_nets[block_idx], datamodule=self.dib_dm)

            # log final training loss, i.e., decodable information
            dib = dib_trainer.logged_metrics["train_loss"]
            network.log(f"dib_{block_idx}", dib)
