import argparse
import copy
from typing import Literal

import torch
from lightning import Callback, Trainer

from datasets import DIBData
from networks import DIBNetwork, MetricNetwork

# defaults for command line arguments
BATCH_SIZE = 400
COMPILE = True
DIB_EPOCHS = 30
EPOCHS = 15
LR = 2e-3
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
        "-c",
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=COMPILE,
        help=(
            "may not work with multiple devices, "
            "see (https://lightning.ai/docs/pytorch/stable/advanced/compile.html#limitations)"
        ),
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="CIFAR10",
        choices=["CIFAR10", "FashionMNIST", "MNIST", "SZT"],
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dib-epochs", default=DIB_EPOCHS, type=int)
    parser.add_argument("--epochs", default=EPOCHS, type=int)
    parser.add_argument("-lr", "--learning-rate", default=LR, type=float)
    parser.add_argument("--loss", default="CrossEntropy", choices=["CrossEntropy"])
    parser.add_argument(
        "-m", "--model", default="CNN", choices=["CIFARNet", "CNN", "ConvNeXt", "MLP", "ResNet"]
    )
    parser.add_argument(
        "-nl",
        "--nonlinearity",
        default="ReLU",
        help=(
            "nonlinearity used in hidden layers of MLP and CNN, "
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


class ComputeNC1(Callback):
    """Compute and log NC metrics at the end of each epoch"""

    def __init__(
        self,
        train_class_counts: torch.Tensor,
        val_class_counts: torch.Tensor,
        num_classes: int,
    ):
        """
        Store metrics for each layer as dict `layer_metrics`.
        Let zˡ be the size of the flattened activations of layer l and
        C the number of classes, then

        - `size(layer_metrics[l]) == (C + zˡ, zˡ)`
        - `layer_metrics[l][:C]` are the per-class activation sums
        - `layer_metrics[l][C:]` is the gram matrix
        """
        self.train_class_counts = train_class_counts
        self.val_class_counts = val_class_counts
        self.num_classes = num_classes
        self.layer_metrics: dict[str, torch.Tensor] = {}

    @torch.inference_mode()
    def _on_batch_end(self, network: MetricNetwork, batch: list[torch.Tensor]):
        """Update NC metrics after each batch"""
        _, targets = batch
        for layer, activations in network.batch_activations.items():
            activations = activations.flatten(start_dim=1)
            if layer not in self.layer_metrics:
                act_size = activations.size(1)
                self.layer_metrics[layer] = torch.zeros(
                    (self.num_classes + act_size, act_size), device=network.device
                )
            self.layer_metrics[layer][: self.num_classes].index_add_(
                dim=0, index=targets, source=activations
            )
            self.layer_metrics[layer][self.num_classes :] += activations.T @ activations

    def on_train_batch_end(
        self,
        _trainer,
        network: MetricNetwork,
        _outputs,
        batch: list[torch.Tensor],
        _batch_idx,
    ):
        self._on_batch_end(network, batch)

    def on_validation_batch_end(
        self,
        _trainer,
        network: MetricNetwork,
        _outputs,
        batch: list[torch.Tensor],
        _batch_idx,
    ):
        self._on_batch_end(network, batch)

    @torch.inference_mode()
    def _on_epoch_end(
        self, network: MetricNetwork, state: str, class_counts: torch.Tensor
    ):
        """Aggregate batched NC metrics and log them"""

        # TODO: come up with smarter way to load-balance
        # synchronise metrics to process 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for joint_metric in self.layer_metrics.values():
                torch.distributed.reduce(joint_metric, dst=0)

        # only compute and log nc on process 0
        if network.global_rank == 0:
            nc: dict[str, torch.Tensor] = {}
            class_counts = class_counts.to(network.device)
            total_count = class_counts.sum()
            for layer, joint_metric in self.layer_metrics.items():
                class_sums = joint_metric[: self.num_classes]
                class_means = class_sums / class_counts.unsqueeze(dim=1)
                global_mean = class_sums.sum(dim=0) / total_count
                centred_means = class_means - global_mean
                between_cov = (centred_means.T @ centred_means) / self.num_classes
                within_cov = (
                    joint_metric[self.num_classes :] / total_count
                    - torch.outer(global_mean, global_mean)
                    - between_cov
                )
                nc[f"{layer}_{state}"] = torch.linalg.lstsq(
                    between_cov.cpu(),  # gelsd only supported on CPU
                    within_cov.cpu(),  # gelsd only supported on CPU
                    driver="gelsd",
                ).solution.trace()
            network.log_dict(nc, rank_zero_only=True)

        # reset NC metrics for next epoch
        self.layer_metrics.clear()

    def on_train_epoch_end(self, _trainer, network: MetricNetwork):
        self._on_epoch_end(network, "train", self.train_class_counts)

    def on_validation_epoch_end(self, _trainer, network: MetricNetwork):
        self._on_epoch_end(network, "val", self.val_class_counts)

    def on_validation_start(self, _trainer, network: MetricNetwork):
        """
        Store and reset training metrics.
        Necessary since validation called before train epoch end
        """
        self.train_batch_activations = copy.deepcopy(network.batch_activations)
        self.train_layer_metrics = copy.deepcopy(self.layer_metrics)
        network.batch_activations.clear()
        self.layer_metrics.clear()

    def on_validation_end(self, _trainer, network: MetricNetwork):
        """Restore training metrics"""
        network.batch_activations = self.train_batch_activations
        self.layer_metrics = self.train_layer_metrics


class ComputeDIB(Callback):
    """Compute and log the DIB metric at the end of each epoch"""

    def __init__(
        self,
        num_decoders_train: int,
        num_decoders_val: int,
        dib_epochs: int,
        dib_dm: DIBData,
        num_devices: int,
        block_indices: list[int],
        no_compile: bool,
    ):
        self.num_decoders = {"train": num_decoders_train, "val": num_decoders_val}
        self.dib_epochs = dib_epochs
        self.dib_dm = dib_dm
        self.num_devices = num_devices
        self.block_indices = block_indices
        self.no_compile = no_compile
        self.dib_nets: dict[str, list[DIBNetwork]] = {"train": [], "val": []}

    def on_fit_start(self, _trainer, network: MetricNetwork):
        """Create DIB networks for each block"""
        for state in ("train", "val"):
            for block_idx in self.block_indices:
                steps_per_epoch = len(getattr(self.dib_dm, f"{state}_dataloader")())
                dib_net = DIBNetwork(
                    *network.get_encoder_decoder(block_idx),
                    self.num_decoders[state],
                    network.optimiser,
                    network.learning_rate,
                    network.num_classes,
                    steps_per_epoch * self.dib_epochs,
                )
                dib_net.compile(
                    disable=self.no_compile,
                    fullgraph=True,
                    options={"max_autotune": True},
                )
                self.dib_nets[state].append(dib_net)

    def _on_epoch_end(
        self,
        trainer: Trainer,
        network: MetricNetwork,
        state: Literal["train", "val"],
        dataloader,
    ):
        """Train the DIB network and log the final training loss"""

        # only train DIB0 (DIB network with non-trainable encoder) once
        not_first = trainer.current_epoch > 0
        for i, block_idx in enumerate(
            self.block_indices[int(not_first) :], start=int(not_first)
        ):
            # update DIB network before training
            encoder, _ = network.get_encoder_decoder(block_idx)
            self.dib_nets[state][i].update_encoder(encoder)
            self.dib_nets[state][i].reset_decoders()

            # train DIB network
            dib_trainer = Trainer(
                devices=self.num_devices,
                max_epochs=self.dib_epochs,
                logger=False,  # don't write (but do store) training losses
                default_root_dir="lightning_logs",
                benchmark=True,
                # deterministic=True, # ignored when benchmark=True
            )
            dib_trainer.fit(self.dib_nets[state][i], dataloader)

            # only log on process 0 since value is the same for all processes
            if network.global_rank == 0:
                # log final training loss, i.e., decodable information
                dib = dib_trainer.logged_metrics["train_loss"]
                network.log(f"dib_{block_idx}_{state}", dib, rank_zero_only=True)

    def on_train_epoch_end(self, trainer: Trainer, network: MetricNetwork):
        self._on_epoch_end(trainer, network, "train", self.dib_dm.train_dataloader())

    def on_validation_epoch_end(self, trainer: Trainer, network: MetricNetwork):
        self._on_epoch_end(trainer, network, "val", self.dib_dm.val_dataloader())

    def on_fit_end(self, _trainer, _network):
        """Clear DIB networks and free memory"""
        self.dib_nets.clear()
