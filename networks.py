from collections import OrderedDict
from copy import deepcopy

import torch
import lightning as L
from torch import nn
from torchvision.models import convnext_tiny, resnet18
from torchvision.models.feature_extraction import create_feature_extractor

from utils import NL_TYPE, LOSS_TYPE, OPTIMISER_TYPE
from datasets import DecoderData


class _Network(L.LightningModule):
    def __init__(
        self, criterion: LOSS_TYPE, optimiser: OPTIMISER_TYPE, logging: bool = True
    ):
        super().__init__()

        # store architecture parameters
        self.criterion = criterion(reduction="none")
        self.optimiser = optimiser
        self.logging = logging

        # final softmax layer
        self.softmax = nn.Softmax(dim=-1)

    def _forward(self, x):
        """Internal forward implementation (sans softmax)"""
        raise NotImplementedError

    def forward(self, x):
        return self.softmax(self._forward(x))

    def training_step(self, batch: list[torch.Tensor]):
        """Compute and log training loss by averaging over classes, (decoders) and then the batch"""
        inputs, targets = batch
        loss: torch.Tensor = self.criterion(self(inputs), targets)
        while loss.dim() > 0:
            loss = loss.mean(dim=-1)
        if self.logging:
            self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return self.optimiser(self.parameters())


class Decoders(_Network):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        num_decoders: int,
        optimiser: OPTIMISER_TYPE,
    ):
        super().__init__(
            criterion=nn.CrossEntropyLoss, optimiser=optimiser, logging=False
        )
        self.encoder = encoder
        self.decoders = nn.ModuleList(deepcopy(decoder) for _ in range(num_decoders))

        # freeze encoder
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _forward(self, x):
        encoded = self.encoder(x)
        return torch.stack([d(encoded) for d in self.decoders], dim=-2)

    def on_train_end(self):
        """Compute and log average loss over all decoders"""
        self.eval()
        with torch.no_grad():
            totals = torch.zeros(2)  # totals = [total_loss, count]
            batch: list[torch.Tensor]
            for batch in self.trainer.train_dataloader:
                batch_size = batch[0].size(0)
                totals[0] += super().training_step(batch) * batch_size
                totals[1] += batch_size
        self.log("v_info", totals[0] / totals[1])

    def configure_optimizers(self):
        return self.optimiser(self.decoders.parameters())


class _MetricNetwork(_Network):
    def __init__(
        self,
        criterion: LOSS_TYPE,
        optimiser: OPTIMISER_TYPE,
        decoder_trainer: L.Trainer,
        decoder_dm: DecoderData,
        num_classes: int,
    ):
        super().__init__(criterion, optimiser)

        # store decoder parameters
        self.decoder_trainer = decoder_trainer
        self.decoder_dm = decoder_dm

        # initial feature extractor
        self.feature_extractor: torch.fx.GraphModule

        # storage for NC metrics
        #   hˡ: size of flattened activations of layer l
        #   layer_metrics[l]: ((num_classes + hˡ) × hˡ)
        #       layer_metrics[l][:num_classes]: class sums of activations (num_classes × hˡ)
        #       layer_metrics[l][num_classes:]: gram matrix (hˡ × hˡ)
        self.num_classes = num_classes
        self.class_counts = torch.zeros(num_classes)
        self.layer_metrics: dict[str, torch.Tensor] = {}

    def _update_feature_extractor(self, new_hooks: dict[str, str]):
        """Update the feature extractor by calling `forward` and setting hooks"""
        out_hooks = new_hooks | {"softmax": "nc_output"}
        self.feature_extractor = create_feature_extractor(self, return_nodes=out_hooks)

    def on_train_batch_start(self, batch: list[torch.Tensor], _batch_idx):
        """Update NC metrics"""
        with torch.no_grad():
            inputs, targets = batch
            self.class_counts += targets.sum(dim=0)
            activations_dict: dict[str, torch.Tensor] = self.feature_extractor(inputs)
            for layer, activations in activations_dict.items():
                activations = activations.flatten(start_dim=1)
                target_and_activations = torch.cat([targets, activations], dim=1)
                joint_metric = target_and_activations.T @ activations
                if layer not in self.layer_metrics:
                    self.layer_metrics[layer] = torch.zeros_like(joint_metric)
                self.layer_metrics[layer] += joint_metric

    def on_train_epoch_end(self):
        """Compute and log NC metrics. Also train decoders."""
        with torch.no_grad():
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
                nc[layer] = (within_cov @ between_cov.pinverse()).trace()
        self.log_dict(nc)

        # reset NC metrics
        self.class_counts.zero_()
        self.layer_metrics.clear()

        # train decoders
        num_decoders = self.decoder_dm.base_expansion.size(1)
        decoders = Decoders(  # TODO: try different block index
            *self.get_encoder_decoder(block_idx=3), num_decoders, self.optimiser
        )
        # self.decoder_trainer.fit(decoders, datamodule=self.decoder_dm)

    def get_encoder_decoder(self, block_idx) -> tuple[nn.Module, nn.Module]:
        """
        Get a copy of the encoder up to start of block `block_idx`
        and the decoder (sans softmax) containing the rest of the network.
        """
        raise NotImplementedError


class MLP(_MetricNetwork):
    def __init__(
        self,
        widths: list[int],
        nonlinearity: NL_TYPE,
        hyperparams: tuple[LOSS_TYPE, OPTIMISER_TYPE, L.Trainer, DecoderData],
    ):
        super().__init__(*hyperparams, num_classes=widths[-1])

        # add layers
        self.flatten = nn.Flatten()
        self.blocks = nn.Sequential(
            *[
                nn.Sequential(
                    OrderedDict([("fc", nn.Linear(*in_out)), ("nl", nonlinearity())])
                )
                for in_out in zip(widths[:-2], widths[1:-1])
            ]
        )
        self.fc = nn.Linear(widths[-2], self.num_classes)

        # update return nodes (output hooks)
        super()._update_feature_extractor(
            {f"blocks.{i}.nl": f"nc_layer_{i}" for i, _ in enumerate(self.blocks)}
        )

    def _forward(self, x):
        return self.fc(self.blocks(self.flatten(x)))

    def get_encoder_decoder(self, block_idx):
        assert block_idx < len(self.blocks), f"MLP only has {len(self.blocks)} blocks"

        encoder = nn.Sequential(self.flatten, *self.blocks[:block_idx])
        decoder = nn.Sequential(*self.blocks[block_idx:], self.fc)
        return encoder, decoder


class ConvNeXt(_MetricNetwork):
    def __init__(
        self,
        in_shape: torch.Size,
        num_classes: int,
        hyperparams: tuple[LOSS_TYPE, OPTIMISER_TYPE, L.Trainer, DecoderData],
    ):
        super().__init__(*hyperparams, num_classes)

        # import torchvision model
        self.convnext = convnext_tiny()

        # replace first and last layer to match input and output shapes
        old_conv1 = self.convnext.features[0][0]
        self.convnext.features[0][0] = nn.Conv2d(
            in_channels=in_shape[0],
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
        )
        self.convnext.classifier[2] = nn.Linear(
            self.convnext.classifier[2].in_features, num_classes
        )

        # update return nodes (output hooks)
        super()._update_feature_extractor(
            {
                f"convnext.features.{2 * i + 1}.{j}.add": f"nc_layer_{i}"
                for i, j in zip(range(4), (2, 2, 8, 2))
            }
        )

    def _forward(self, x):
        return self.convnext(x)

    def get_encoder_decoder(self, block_idx):
        num_blocks = len(self.convnext.features) // 2
        assert block_idx < num_blocks, f"ConvNeXt only has {num_blocks} blocks"

        encoder = self.convnext.features[: 2 * block_idx]
        # Fixed sequential construction - don't use append
        decoder = nn.Sequential(
            *self.convnext.features[2 * block_idx :],
            self.convnext.avgpool,
            self.convnext.classifier,
        )
        return encoder, decoder


class ResNet(_MetricNetwork):
    def __init__(
        self,
        in_shape: torch.Size,
        num_classes: int,
        hyperparams: tuple[LOSS_TYPE, OPTIMISER_TYPE, L.Trainer, DecoderData],
    ):
        super().__init__(*hyperparams, num_classes)

        # import torchvision model
        self.resnet = resnet18()

        # replace first and last layer to match input and output shapes
        old_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            in_channels=in_shape[0],
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=old_conv1.bias is not None,
        )
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

        # update return nodes (output hooks)
        super()._update_feature_extractor(
            {f"resnet.layer{i + 1}.1.relu_1": f"nc_layer_{i}" for i in range(4)}
        )

    def _forward(self, x):
        return self.resnet(x)

    def get_encoder_decoder(self, block_idx):
        assert block_idx < 5, "ResNet only has 4 blocks"

        encoder = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            *[getattr(self.resnet, f"layer{i}") for i in range(1, block_idx + 1)],
        )
        decoder = nn.Sequential(
            *[getattr(self.resnet, f"layer{i}") for i in range(block_idx + 1, 5)],
            self.resnet.avgpool,
            self.resnet.fc,
        )
        return encoder, decoder
