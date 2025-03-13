from collections import OrderedDict
from copy import deepcopy

from lightning import LightningModule, Trainer
import torch
from torch import nn
from torch.utils.hooks import RemovableHandle
from torchvision.models import convnext_tiny, resnet18

from datasets import DecoderData
from utils import LOSS_TYPE, NL_TYPE, OPTIMISER_TYPE


HPARAM_TYPE = tuple[LOSS_TYPE, OPTIMISER_TYPE, int, int, DecoderData]


class _Network(LightningModule):
    def __init__(
        self, criterion: LOSS_TYPE, optimiser: OPTIMISER_TYPE, logging: bool = True
    ):
        super().__init__()

        # store architecture parameters
        self.criterion = criterion
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
        """Compute and log average training loss"""
        inputs, targets = batch
        loss: torch.Tensor = self.criterion()(self(inputs), targets)
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
        true_epoch: int,
    ):
        super().__init__(
            criterion=nn.CrossEntropyLoss, optimiser=optimiser, logging=False
        )
        self.encoder = encoder
        self.decoders = nn.ModuleList(deepcopy(decoder) for _ in range(num_decoders))
        self.true_epoch = true_epoch

    def _forward(self, x):
        encoded = self.encoder(x)
        return torch.stack([d(encoded) for d in self.decoders], dim=-1)

    def on_train_end(self):
        """Compute and log final average loss"""
        self.eval()
        with torch.no_grad():
            totals = torch.zeros(2)  # totals = [total_loss, count]
            batch: list[torch.Tensor]
            for batch in self.trainer.train_dataloader:
                batch_size = batch[0].size(0)
                totals[0] += super().training_step(batch) * batch_size
                totals[1] += batch_size
        self.logger.log_metrics(
            {"epoch": self.true_epoch, "v_info": (totals[0] / totals[1]).item()},
            step=self.global_step,
        )

    def configure_optimizers(self):
        return self.optimiser(self.decoders.parameters())

    def freeze_encoder(self):
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        self.encoder.train()
        for param in self.encoder.parameters():
            param.requires_grad = True


class _MetricNetwork(_Network):
    def __init__(
        self,
        criterion: LOSS_TYPE,
        optimiser: OPTIMISER_TYPE,
        num_decoders: int,
        decoder_epochs: int,
        decoder_dm: DecoderData,
        num_classes: int,
    ):
        super().__init__(criterion, optimiser)

        # store decoder parameters
        self.num_decoders = num_decoders
        self.decoder_epochs = decoder_epochs
        self.decoder_dm = decoder_dm

        # storing handles to hooks not strictly necessary, but good practice
        self.hook_handles: dict[str, RemovableHandle] = {}

        # storage for NC metrics
        #   hˡ: size of flattened activations of layer l
        #   layer_metrics[l]: ((num_classes + hˡ) × hˡ)
        #       layer_metrics[l][:num_classes]: class sums of activations (num_classes × hˡ)
        #       layer_metrics[l][num_classes:]: gram matrix (hˡ × hˡ)
        self.num_classes = num_classes
        self.class_counts = torch.zeros(num_classes)
        self.layer_outputs: dict[str, torch.Tensor] = {}
        self.layer_metrics: dict[str, torch.Tensor] = {}

    def _register_hooks(self, new_hooks: dict[str, nn.Module]):
        """Register forward hooks for NC metrics"""
        # remove old hooks (there shouldn't be any)
        for handle in self.hook_handles.values():
            handle.remove()
        self.hook_handles.clear()

        # get_hook: layer_name -> (hook: module, args, output -> None)
        def get_hook(name):
            def hook(_module, _args, output):
                self.layer_outputs[name] = output

            return hook

        # register new hooks
        new_hooks |= {"nc_output": self.softmax}
        for name, module in new_hooks.items():
            self.hook_handles[name] = module.register_forward_hook(get_hook(name))

    def on_train_batch_end(self, _outputs, batch: list[torch.Tensor], _batch_idx):
        """Update NC metrics"""
        with torch.no_grad():
            _, targets = batch
            self.class_counts += targets.sum(dim=0)
            for layer, activations in self.layer_outputs.items():
                activations = activations.flatten(start_dim=1)
                if layer not in self.layer_metrics:
                    h_l = activations.size(1)
                    self.layer_metrics[layer] = torch.zeros(
                        (self.num_classes + h_l, h_l)
                    )
                self.layer_metrics[layer][: self.num_classes] += targets.T @ activations
                self.layer_metrics[layer][self.num_classes :] += (
                    activations.T @ activations
                )

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
                nc[layer] = torch.linalg.lstsq(
                    between_cov,
                    within_cov,
                    rcond=1e-15,  # default value of pinverse
                ).solution.trace()

        self.log_dict(nc)

        # reset NC metrics
        self.class_counts.zero_()
        self.layer_metrics.clear()

        # train decoders
        decoders = Decoders(  # TODO: try different block index
            *self.get_encoder_decoder(block_idx=3),
            self.num_decoders,
            self.optimiser,
            self.current_epoch,
        )
        decoders.freeze_encoder()
        Trainer(devices=5, max_epochs=self.decoder_epochs, logger=self.logger).fit(
            decoders, datamodule=self.decoder_dm
        )
        decoders.unfreeze_encoder()

    def get_encoder_decoder(self, block_idx) -> tuple[nn.Module, nn.Module]:
        """
        Get a copy of the encoder up to start of block `block_idx`
        and the decoder (sans softmax) containing the rest of the network.
        """
        raise NotImplementedError


class MLP(_MetricNetwork):
    def __init__(
        self, widths: list[int], nonlinearity: NL_TYPE, hyperparams: HPARAM_TYPE
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
        super()._register_hooks(
            {f"nc_layer_{i}": block.nl for i, block in enumerate(self.blocks)}
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
        self, in_shape: torch.Size, num_classes: int, hyperparams: HPARAM_TYPE
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
        new_hooks = {}
        for i, j in zip(range(4), (2, 2, 8, 2)):
            new_hooks[f"nc_layer_{i}"] = self.convnext.features[2 * i + 1][j].add
        super()._register_hooks(new_hooks)

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
        self, in_shape: torch.Size, num_classes: int, hyperparams: HPARAM_TYPE
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
        new_hooks = {}
        for i in range(4):
            layer = getattr(self.resnet, f"layer{i + 1}")
            new_hooks[f"nc_layer_{i}"] = layer[1].relu
        super()._register_hooks(new_hooks)

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
