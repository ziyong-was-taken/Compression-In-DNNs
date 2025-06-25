import argparse
import math
from copy import deepcopy

import torch
from lightning import Callback, Trainer

from datasets import DataModule, DIBData
from networks import MetricNetwork, VInfoNet

# defaults for command line arguments
BATCH_SIZE = 400
COMPILE = True
DATASET = "CIFAR10"
DIB_EPOCHS = 50
EPOCHS = 20
LR = 2e-3
MODEL = "CIFARNet"
NUM_DEVICES = 1
OPTIMISER = "AdamW"
PRECISION = "bf16-true"
TUNE = False

# type aliases
DIB_PARAM_TYPE = tuple[int, int]  # (dib_epochs, num_devices)


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
    -nl,  --nonlinearity: nonlinearity used in hidden layers of MLP, MNISTNet
          --num-devices: number of devices used to train the DIB network
    -opt, --optimiser: optimiser to use
    -p,   --precision: precision to use for training
          --seed: seed for random number generation
          --subset-idx: index of (stratified) training subset to use
          --tune: tune the learning rate
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
        default=DATASET,
        choices=["CIFAR10", "FashionMNIST", "MNIST", "SZT"],
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dib-epochs", default=DIB_EPOCHS, type=int)
    parser.add_argument("--epochs", default=EPOCHS, type=int)
    parser.add_argument("-lr", "--learning-rate", default=LR, type=float)
    parser.add_argument(
        "--loss", default="CrossEntropy", choices=["CrossEntropy"]
    )
    parser.add_argument(
        "-m",
        "--model",
        default=MODEL,
        choices=["CIFARNet", "ConvNeXt", "MLP", "MNISTNet", "ResNet"],
    )
    parser.add_argument(
        "-nl",
        "--nonlinearity",
        default="ReLU",
        help=(
            "nonlinearity used in hidden layers of MLP, MNISTNet, and CIFARNet, "
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
        default=OPTIMISER,
        choices=["AdamW", "Adam", "SGD"],
    )
    parser.add_argument(
        "-p",
        "--precision",
        default=PRECISION,
        choices=["bf16-true", "32-true"],
        help=(
            "precision to use for training, "
            "see https://lightning.ai/docs/pytorch/stable/common/precision.html "
            "note that the NC metric will always be computed in 32-bit precision"
        ),
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--subset-idx",
        default=0,
        type=int,
        help="index of (stratified) training subset to use",
    )
    parser.add_argument("--tune", action=argparse.BooleanOptionalAction, default=TUNE)
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


class ComputeNC(Callback):
    """Compute and log NC metrics at the end of each epoch"""

    def __init__(
        self,
        train_class_counts: torch.Tensor,
        val_class_counts: torch.Tensor,
        num_classes: int,
    ):
        """
        Store per-class activation sums for each layer as dict `all_class_sums`.
        `all_class_sums[l]` is of size (C, zˡ) where C is the number of classes and
        zˡ is the size of the flattened activations of layer l.
        """
        self.train_class_counts = train_class_counts
        self.val_class_counts = val_class_counts
        self.num_classes = num_classes
        self.all_class_sums: dict[str, torch.Tensor] = {}

    @torch.inference_mode()
    def _on_batch_end(self, network: MetricNetwork, batch: list[torch.Tensor]):
        """Update NC metrics after each batch"""
        _, targets = batch
        for layer, activations in network.batch_activations.items():
            if layer not in self.all_class_sums:
                self.all_class_sums[layer] = activations.new_zeros(
                    (self.num_classes, activations.size(1))
                )
            self.all_class_sums[layer].index_add_(
                dim=0, index=targets, source=activations
            )
        network.batch_activations.clear()

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
        self,
        network: MetricNetwork,
        state: str,
        class_counts: torch.Tensor,
        dataloader,
    ):
        """Aggregate batched NC metrics and log them"""

        # synchronise metrics across processes
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for class_sums in self.all_class_sums.values():
                torch.distributed.all_reduce(class_sums)
        class_counts = class_counts.to(network.device)

        class_means: dict[str, torch.Tensor] = {}
        centred_means: dict[str, torch.Tensor] = {}
        gram_matrices = []

        # compute means and gram matrices
        total_count = class_counts.sum()
        for i, (layer, class_sums) in enumerate(self.all_class_sums.items()):
            class_means[layer] = class_sums / class_counts.unsqueeze(
                dim=1
            )  # [𝛍_1^l ⋯ 𝛍_C^l]^⊤: (C, zˡ)
            global_mean = class_sums.sum(dim=0) / total_count  # ̄𝛍ˡ: (zˡ)
            centred_means[layer] = class_means[layer] - global_mean  # (𝐌ˡ)^⊤: (C, zˡ)
            gram_matrices.append(
                centred_means[layer] @ centred_means[layer].T
            )  # (𝐌ˡ)^⊤ 𝐌ˡ: (C, C), recall: Σ_B^l = 1/C 𝐌ˡ(𝐌ˡ)^⊤

        # compute eigendecomposition of (𝐌ˡ)^⊤(𝐌ˡ)
        eigvals, eigvecs = torch.linalg.eigh(
            torch.stack(gram_matrices, dim=0)
        )  # (n_layers, C), (n_layers, C, C)

        # second pass
        nc = {f"{layer}_{state}": 0 for layer in self.all_class_sums.keys()}
        for batch in dataloader:
            inputs, targets = network.transfer_batch_to_device(batch, network.device, 0)

            # compute activations
            with torch.autocast(device_type=network.device.type, dtype=network.dtype):
                network(inputs)

            # aggregate second pass
            for i, (layer, activations) in enumerate(network.batch_activations.items()):
                # filter out eigenvalues and vectors close to zero for stability
                mask = eigvals[i] > 1e-5 * eigvals[i].max()

                matrix_prod = (
                    (activations - class_means[layer][targets]) @ centred_means[layer].T
                ) @ eigvecs[i, :, mask]  # (𝐡_c^l - 𝛍_c^l)^⊤ 𝐌ˡ 𝐕: (batch, r)

                nc[f"{layer}_{state}"] += ((matrix_prod / eigvals[i, mask]) ** 2).sum()

        # scale by C/N
        nc = {key: value * self.num_classes / total_count for key, value in nc.items()}

        # sum metrics across processes again
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for nc_value in nc.values():
                torch.distributed.all_reduce(nc_value)

        # only log on process 0 since value is the same for all processes
        if network.global_rank == 0:
            network.log_dict(nc, rank_zero_only=True)

        # reset NC metrics for next epoch
        self.all_class_sums.clear()

    def on_train_epoch_end(self, trainer: Trainer, network: MetricNetwork):
        self._on_epoch_end(
            network, "train", self.train_class_counts, trainer.train_dataloader
        )

    def on_validation_epoch_end(self, trainer: Trainer, network: MetricNetwork):
        self._on_epoch_end(
            network, "val", self.val_class_counts, trainer.val_dataloaders
        )

    def on_validation_start(self, _trainer, network: MetricNetwork):
        """
        Store training NC metrics and reset them before validation.
        Necessary since validation is done before the end of each training epoch
        (but after all training batches).
        """
        self.train_class_sums = deepcopy(self.all_class_sums)
        self.all_class_sums.clear()

    def on_validation_end(self, _trainer, network: MetricNetwork):
        """Restore training metrics"""
        self.all_class_sums = self.train_class_sums


class _ComputeVInfo(Callback):
    """Compute and log the 𝒱-information at the end of each epoch"""

    def __init__(
        self,
        dm: DataModule | DIBData,
        epochs: int,
        num_devices: int,
        name: str,
        *block_indices: int,
    ):
        self.dm = dm
        self.name = name
        self.v_info_epochs = epochs
        self.num_devices = num_devices
        self.block_indices = block_indices

    def on_train_start(self, _trainer, network: MetricNetwork):
        # default to all blocks if no block indices were given
        if not self.block_indices:
            self.block_indices = list(range(network.num_blocks + 1))

        # create 𝒱-information networks for each block for both datasets
        hyperparams = tuple(network.hparams_initial.values())
        self.v_info_nets = {
            dataset: [
                VInfoNet(
                    *network.get_encoder_decoder(block_idx),
                    self.dm.num_decoders[dataset],
                    hyperparams,
                )
                for block_idx in self.block_indices
            ]
            for dataset in ("train", "val")
        }

    def on_train_epoch_end(self, trainer: Trainer, network: MetricNetwork):
        """
        Train the 𝒱-information networks on both the training and validation set
        and log the final training loss for both sets.
        """
        for dataset in ("train", "val"):
            # only train 𝒱-information network with non-trainable encoder once
            not_first = trainer.current_epoch > 0
            for i, block_idx in enumerate(
                self.block_indices[int(not_first) :], start=int(not_first)
            ):
                # update 𝒱-information network before training
                encoder, _ = network.get_encoder_decoder(block_idx)
                self.v_info_nets[dataset][i].update_encoder(encoder)
                self.v_info_nets[dataset][i].reset_decoders()

                # train 𝒱-information network
                self.dm.setup("fit")
                v_info_trainer = Trainer(
                    devices=self.num_devices,
                    precision=trainer.precision,
                    max_epochs=self.v_info_epochs,
                    logger=False,  # don't write (but do store) training losses
                    default_root_dir="lightning_logs",
                    benchmark=True,
                    # deterministic=True, # ignored when benchmark=True
                )
                v_info_trainer.fit(
                    self.v_info_nets[dataset][i],
                    getattr(self.dm, f"{dataset}_dataloader")(),
                )

                # only log on process 0 since value is the same for all processes
                if network.global_rank == 0:
                    # compute and log 𝒱-information in bits
                    v_info = math.log2(
                        self.dm.num_classes
                    ) - v_info_trainer.logged_metrics["train_loss"] / math.log(2)
                    network.log(
                        f"{self.name}_{block_idx}_{dataset}",
                        v_info,
                        rank_zero_only=True,
                    )

    def on_train_end(self, _trainer, _network):
        """Clear DIB networks and free memory"""
        self.v_info_nets.clear()


class ComputeVSuff(_ComputeVInfo):
    """Compute and log the sufficiency term I_𝒱ˡ(𝗵ˡ → y) at the end of each epoch"""

    def __init__(self, dm: DataModule, epochs: int, num_devices: int):
        super().__init__(dm, epochs, num_devices, "vsuff")


class ComputeVMin(_ComputeVInfo):
    """Compute and log the minimality term I_𝒱ˡ(𝗵ˡ → Dec(𝘅, 𝒴)) at the end of each epoch"""

    def __init__(self, dib_dm: DIBData, dib_epochs: int, num_devices: int):
        super().__init__(dib_dm, dib_epochs, num_devices, "vmin")
