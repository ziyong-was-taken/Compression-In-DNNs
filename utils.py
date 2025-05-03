import argparse
import math

import torch
from lightning import Callback, Trainer

from datasets import DIBData
from networks import DIBNetwork, MetricNetwork


# defaults for command line arguments
BATCH_SIZE = 240
COMPILE = True
DIB_EPOCHS = 30
EPOCHS = 5
LR = 1e-3
NUM_DEVICES = 1


class WideHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Formatter which shows defaults"""

    def __init__(
        self, prog, indent_increment=2, max_help_position=55, width=None
    ) -> None:
        super().__init__(prog, indent_increment, max_help_position, width)


class CustomArgParser(argparse.ArgumentParser):
    """Custom argument parser that uses a custom help formatter"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, formatter_class=WideHelpFormatter, **kwargs)

    def add_argument(self, *args, default=None, help=None, **kwargs):
        kwargs["default"] = default
        kwargs["help"] = (
            help if (help is not None or default is None) else "(default: %(default)s)"
        )
        super().add_argument(*args, **kwargs)


def get_args():
    """
    Parse command line arguments:

    -b,   --batch-size: batch size for training
    -c,   --compile: compile the model
    -d,   --dataset: dataset to use
          --data-dir: directory containing the dataset
          --dib-epochs: number of epochs to train the DIB network
          --epochs: number of epochs to train the model
    -lr,  --learning-rate: learning rate for the optimiser
          --loss: loss function to use
    -m,   --model: model to use
    -nl,  --nonlinearity: nonlinearity used in hidden layers of MLP, CNN
          --num-devices: number of devices used to train the DIB network
    -opt, --optimiser: optimiser to use
          --seed: seed for random number generation
    -w,   --widths: widths of hidden layers of MLP
    """

    parser = CustomArgParser()
    parser.add_argument("-b", "--batch-size", default=BATCH_SIZE, type=int)
    parser.add_argument(
        "-c", "--compile", action=argparse.BooleanOptionalAction, default=COMPILE
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="MNIST",
        choices=["CIFAR10", "FashionMNIST", "MNIST", "SZT"],
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dib-epochs", default=DIB_EPOCHS, type=int)
    parser.add_argument("--epochs", default=EPOCHS, type=int)
    parser.add_argument("-lr", "--learning-rate", default=LR, type=float)
    parser.add_argument("--loss", default="CrossEntropy", choices=["CrossEntropy"])
    parser.add_argument(
        "-m", "--model", default="CNN", choices=["CNN", "ConvNeXt", "MLP", "ResNet"]
    )
    parser.add_argument(
        "-nl",
        "--nonlinearity",
        default="ReLU",
        help=(
            "nonlinearity used in hidden layers of MLP, "
            "has no effect on ConvNeXt and ResNet (yet)"
        ),
    )
    parser.add_argument(
        "--num-devices",
        default=NUM_DEVICES,
        type=int,
        help="number of devices used to train the DIB network",
    )
    parser.add_argument(
        "-opt",
        "--optimiser",
        default="AdamW",
        choices=["AdamW", "Adam", "SGD"],
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "-w",
        "--widths",
        nargs="+",
        default=[10, 7, 5, 4, 3],  # same model as Schwartz-Ziv & Tishby (2017)
        type=int,
        help="widths of hidden layers of MLP, has no effect on other models (yet)",
        metavar="WIDTH",
    )
    return parser.parse_args()


def total_steps(len_dataset: int, batch_size: int, num_epochs: int):
    return math.ceil(len_dataset // batch_size) * num_epochs


def base_expand(labels: torch.Tensor, num_classes: int):
    r"""
    Generate random relabeling `N` of the data where `N[i,:]` are
    the new labels of the `i`th sample.
    Each sample obtains ⌈log_`num_classes`(max{|X_y| : y ∈ Y})⌉ new labels,
    where X_y = {l ∈ `labels` : l = y} and Y = {0,…,`num_classes`-1}.
    Based on Algorithm 1 of "Learning Optimal Representations with
    the Decodable Information Bottleneck".
    """
    assert labels.dim() == 1, "labels must be 1D"
    assert torch.max(labels) < num_classes, "labels ⊈ {0,…,num_classes}"

    # compute ⌈log_{|Y|}(max{|X_y| : y ∈ Y})⌉
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

    def __init__(self, class_counts: torch.Tensor):
        """
        Store metrics for each layer as dict `layer_metrics`.
        Let zˡ be the size of the flattened activations of layer l and
        C the number of classes, then

        - `size(layer_metrics[l]) == (C + zˡ, zˡ)`
        - `layer_metrics[l][:C]` are the per-class activation sums
        - `layer_metrics[l][C:]` is the gram matrix
        """
        self.class_counts = class_counts
        self.num_classes = class_counts.size(0)
        self.layer_metrics: dict[str, torch.Tensor] = {}

    @torch.inference_mode()
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
        num_classes = self.class_counts.size(0)
        for layer, activations in network.batch_activations.items():
            activations = activations.flatten(start_dim=1)
            if layer not in self.layer_metrics:
                act_size = activations.size(1)
                self.layer_metrics[layer] = torch.zeros(
                    (num_classes + act_size, act_size), device=network.device
                )
            self.layer_metrics[layer][:num_classes].index_add_(
                dim=0, index=targets, source=activations
            )
            self.layer_metrics[layer][num_classes:] += activations.T @ activations

    @torch.inference_mode()
    def on_train_epoch_end(self, _trainer, network: MetricNetwork):
        """Aggregate batched NC metrics and log them"""

        # TODO: come up with smarter way to load-balance
        # synchronise metrics to process 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for joint_metric in self.layer_metrics.values():
                torch.distributed.reduce(joint_metric, dst=0)

        # only compute and log nc on process 0
        if network.global_rank == 0:
            self.class_counts = self.class_counts.to(network.device)
            num_classes = self.class_counts.size(0)
            nc: dict[str, torch.Tensor] = {}
            total_count = self.class_counts.sum()
            for layer, joint_metric in self.layer_metrics.items():
                class_sums = joint_metric[:num_classes]
                class_means = class_sums / self.class_counts.unsqueeze(dim=1)
                global_mean = class_sums.sum(dim=0) / total_count
                centred_means = class_means - global_mean
                between_cov = (centred_means.T @ centred_means) / num_classes
                within_cov = (
                    joint_metric[num_classes:] / total_count
                    - torch.outer(global_mean, global_mean)
                    - between_cov
                )
                nc[layer] = torch.linalg.lstsq(
                    between_cov.cpu(),  # gelsd only supported on CPU
                    within_cov.cpu(),  # gelsd only supported on CPU
                    driver="gelsd",
                ).solution.trace()
            network.log_dict(nc, rank_zero_only=True)

        # reset NC metrics for next epoch (class counts don't change)
        self.layer_metrics.clear()

    # TODO: account for train class counts ≠ val class counts
    # NOTE: validation called before train epoch end
    def on_validation_batch_end(
        self,
        _trainer,
        network: MetricNetwork,
        _outputs,
        batch: list[torch.Tensor],
        _batch_idx,
    ):
        pass

    def on_validation_epoch_end(self, _trainer, network: MetricNetwork):
        pass

    # TODO: account for train class counts ≠ test class counts
    def on_test_batch_end(
        self,
        _trainer,
        network: MetricNetwork,
        _outputs,
        batch: list[torch.Tensor],
        _batch_idx,
    ):
        pass

    def on_test_epoch_end(self, _trainer, network: MetricNetwork):
        pass


class ComputeDIB(Callback):
    """Compute and log the DIB metric at the end of each epoch"""

    def __init__(
        self,
        num_decoders: int,
        dib_epochs: int,
        dib_dm: DIBData,
        num_devices: int,
        block_indices: list[int],
        no_compile: bool,
    ):
        self.num_decoders = num_decoders
        self.dib_epochs = dib_epochs
        self.dib_dm = dib_dm
        self.num_devices = num_devices
        self.block_indices = block_indices
        self.no_compile = no_compile
        self.dib_nets: list[DIBNetwork] = []

    def on_train_start(self, _trainer, network: MetricNetwork):
        """Create DIB networks for each block"""
        for block_idx in self.block_indices:
            dib_net = DIBNetwork(
                *network.get_encoder_decoder(block_idx),
                self.num_decoders,
                network.optimiser,
                network.learning_rate,  # TODO: find optimal learning rate for DIB
                network.num_classes,
                total_steps(
                    len(self.dib_dm.train), self.dib_dm.batch_size, self.dib_epochs
                ),
            )
            dib_net.compile(disable=self.no_compile)
            self.dib_nets.append(dib_net)

    def on_train_epoch_end(self, trainer: Trainer, network: MetricNetwork):
        """Train the DIB network and log the final training loss"""

        # only train DIB0 (DIB network with non-trainable encoder) once
        not_first = trainer.current_epoch > 0
        for i, block_idx in enumerate(
            self.block_indices[int(not_first) :], start=int(not_first)
        ):
            # update encoder
            encoder, _ = network.get_encoder_decoder(block_idx)
            self.dib_nets[i].update_encoder(encoder)

            # train DIB network
            dib_trainer = Trainer(
                devices=self.num_devices,
                max_epochs=self.dib_epochs,
                logger=False,  # don't write (but do store) training losses
                default_root_dir="lightning_logs",
                benchmark=True,
                # deterministic=True, # ignored when benchmark=True
            )
            dib_trainer.fit(self.dib_nets[i], datamodule=self.dib_dm)

            # only log on process 0 since value is the same for all processes
            if network.global_rank == 0:
                # log final training loss, i.e., decodable information
                dib = dib_trainer.logged_metrics["train_loss"]
                network.log(f"dib_{block_idx}", dib, rank_zero_only=True)

            # free memory
            del dib_trainer

    def on_train_end(self, _trainer, _network):
        self.dib_nets.clear()

    def on_validation_epoch_end(self, trainer: Trainer, network: MetricNetwork):
        self.on_train_epoch_end(trainer, network)

    def on_test_start(self, _trainer, network: MetricNetwork):
        self.on_train_start(_trainer, network)

    def on_test_end(self, trainer: Trainer, network: MetricNetwork):
        self.on_train_epoch_end(trainer, network)
