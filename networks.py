from collections import OrderedDict
from copy import deepcopy

import torch
from lightning import LightningModule
from torch import nn, optim
from torchmetrics.functional.classification import multiclass_accuracy
from torchvision.models import convnext_tiny, resnet18

NL_TYPE = type[nn.ReLU | nn.Tanh]
LOSS_TYPE = type[nn.CrossEntropyLoss]
OPT_TYPE = type[optim.AdamW | optim.Adam | optim.SGD]
HPARAM_TYPE = tuple[
    LOSS_TYPE,  # loss function
    OPT_TYPE,  # optimiser
    float,  # learning rate
    int,  # number of classes
    bool,  # if the model should be compiled
]


class _Network(LightningModule):
    """
    The blueprint for a basic network.
    Simply implement `forward` and the blueprint handles the rest.
    """

    def __init__(
        self,
        criterion: LOSS_TYPE,
        optimiser: OPT_TYPE,
        learning_rate: float,
        num_classes: int,
        no_compile: bool,
        ignore: list[str] = [],  # ignored hyperparameters
    ):
        super().__init__()
        self.save_hyperparameters(ignore=ignore)

    def configure_model(self):
        self = self.to(memory_format=torch.channels_last)
        self.compile(
            disable=self.hparams_initial["no_compile"],
            fullgraph=True,
            mode="max-autotune",
        )

    def forward(self, x):
        raise NotImplementedError

    def training_step(self, batch: list[torch.Tensor]):
        """Compute and log average training loss"""
        inputs, targets = batch
        outputs: torch.Tensor = self(inputs)
        loss = self.hparams_initial["criterion"]()(outputs, targets)
        self.log("train_loss", loss, sync_dist=True, on_step=False, on_epoch=True)

        with torch.inference_mode():
            acc = multiclass_accuracy(
                outputs.argmax(dim=1),
                targets,
                num_classes=self.hparams_initial["num_classes"],
            )
            self.log("train_acc", acc, sync_dist=True, on_step=False, on_epoch=True)

        return loss

    def _opt_parameters(self):
        """Parameters to optimise"""
        return self.parameters()

    def configure_optimizers(self):
        optimiser = self.hparams_initial["optimiser"](
            self._opt_parameters(), lr=self.hparams_initial["learning_rate"], fused=True
        )
        return optimiser


class DIBNetwork(_Network):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        num_decoders: int,
        hyperparams: HPARAM_TYPE,
    ):
        """
        Create a network with multiple copies of `decoder` connected to a frozen copy of `encoder`.
        ```plaintext
                   decoder →
                  / 
        → encoder -   ⋮     → 
                  \\
                   decoder →
        ```
        Before training, all decoders are reset.
        """
        super().__init__(nn.CrossEntropyLoss, *hyperparams[1:], ["encoder", "decoder"])

        # copy encoder to ensure correct device placement of parameters
        self.encoder = deepcopy(encoder).requires_grad_(False).eval()
        self.decoders = nn.ModuleList(deepcopy(decoder) for _ in range(num_decoders))

    def update_encoder(self, encoder: nn.Module):
        """Update the parameters of the encoder"""
        self.encoder.load_state_dict(encoder.state_dict())

    def reset_decoders(self):
        """
        Reset decoder weights.
        See https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819
        """

        @torch.no_grad()
        def reset_weights(m: nn.Module):
            reset_parameters = getattr(
                m, "reset_parameters", getattr(m, "_reset_parameters", None)
            )
            if callable(reset_parameters):
                reset_parameters()

        self.decoders.apply(reset_weights)

    def forward(self, x):
        encoded = self.encoder(x)
        outputs = [d(encoded) for d in self.decoders]
        return torch.stack(outputs, dim=-1)

    def _opt_parameters(self):
        return self.decoders.parameters()


class MetricNetwork(_Network):
    """
    A blueprint which extends `_Network` with helper functions to compute NC1 and DIB
    """

    def __init__(self, hyperparams: HPARAM_TYPE, ignore: list[str] = []):
        super().__init__(*hyperparams, ignore=["hyperparams", "ignore"] + ignore)
        self.num_blocks: int
        self.batch_activations: dict[str, torch.Tensor] = {}

    def _register_hooks(self, new_hooks: dict[str, nn.Module]):
        """Register forward hooks for NC metrics"""

        # get_hook: layer_name -> (hook: module, args, output -> None)
        def get_hook(name):
            def hook(_module, _args, output: torch.Tensor):
                self.batch_activations[name] = (
                    output.detach().flatten(start_dim=1).float()
                )

            return hook

        # register new hooks
        for name, module in new_hooks.items():
            module.register_forward_hook(get_hook(name))

    def validation_step(self, batch: list[torch.Tensor]):
        """Compute and log average validation loss"""
        inputs, targets = batch
        outputs: torch.Tensor = self(inputs)
        loss = self.hparams_initial["criterion"]()(outputs, targets)
        acc = multiclass_accuracy(
            outputs.argmax(dim=1),
            targets,
            num_classes=self.hparams_initial["num_classes"],
        )
        self.log("val_loss", loss, sync_dist=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, sync_dist=True, on_step=False, on_epoch=True)

    def _check_encoder_blocks(self, encoder_blocks: int):
        assert encoder_blocks <= self.num_blocks, (
            f"{type(self).__name__} only has {self.num_blocks} blocks"
        )

    def get_encoder_decoder(self, encoder_blocks: int) -> tuple[nn.Module, nn.Module]:
        """
        Get a copy of the encoder containing `encoder_blocks` blocks
        and the decoder (sans softmax) containing the rest of the network.
        """
        raise NotImplementedError


class MLP(MetricNetwork):
    def __init__(
        self,
        widths: list[int],
        nonlinearity: NL_TYPE,
        hyperparams: HPARAM_TYPE,
    ):
        super().__init__(hyperparams, ["widths", "nonlinearity"])

        # add layers
        self.flatten = nn.Flatten()
        self.blocks = nn.Sequential(
            *[
                nn.Sequential(
                    OrderedDict({"fc": nn.Linear(*in_out), "nl": nonlinearity()})
                )
                for in_out in zip(widths[:-1], widths[1:])
            ]
        )
        self.fc = nn.Linear(widths[-1], hyperparams[3])

        # update return nodes (output hooks)
        self._register_hooks(
            {f"nc_layer_{i + 1}": block for i, block in enumerate(self.blocks)}
        )
        self.num_blocks = len(self.blocks)

    def forward(self, x):
        return self.fc(self.blocks(self.flatten(x)))

    def get_encoder_decoder(self, encoder_blocks):
        self._check_encoder_blocks(encoder_blocks)
        encoder = nn.Sequential(self.flatten, *self.blocks[:encoder_blocks])
        decoder = nn.Sequential(*self.blocks[encoder_blocks:], self.fc)
        return encoder, decoder


class _ConvPoolActivation(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, nonlinearity: NL_TYPE):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 5, padding="same")
        self.bn = nn.BatchNorm2d(out_channels, track_running_stats=False)
        self.pool = nn.MaxPool2d(3)
        self.nl = nonlinearity()

    def forward(self, x):
        return self.nl(self.pool(self.bn(self.conv(x))))


class MNISTNet(MetricNetwork):
    """
    Based on the current 94% CIFAR-10 speedrun world record holder by Keller Jordan
    (https://github.com/KellerJordan/cifar10-airbench) and
    Tuomas Oikarinen's performance-focussed (both in speed and accuracy) CNN.
    (https://github.com/tuomaso/train_mnist_fast)
    """

    def __init__(
        self, in_shape: torch.Size, nonlinearity: NL_TYPE, hyperparams: HPARAM_TYPE
    ):
        super().__init__(hyperparams, ["in_shape", "nonlinearity"])

        channels = (in_shape[0], 64, 128, 256)
        self.blocks = nn.Sequential(
            *(
                _ConvPoolActivation(*in_out_kernels, nonlinearity)
                for in_out_kernels in zip(channels[:-1], channels[1:])
            )
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(channels[-1], hyperparams[3]))

        # update return nodes (output hooks)
        self._register_hooks(
            {f"nc_layer_{i + 1}": block for i, block in enumerate(self.blocks)}
        )
        self.num_blocks = len(self.blocks)

    def forward(self, x):
        return self.head(self.blocks(x))

    def get_encoder_decoder(self, encoder_blocks):
        super()._check_encoder_blocks(encoder_blocks)
        encoder = nn.Sequential(*self.blocks[:encoder_blocks])
        decoder = nn.Sequential(*self.blocks[encoder_blocks:], self.head)
        return encoder, decoder


class _Block(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, nonlinearity: NL_TYPE
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding="same", bias=False
        )
        self.bn = nn.BatchNorm2d(in_channels, track_running_stats=False)
        self.conv2 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding="same", bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels, track_running_stats=False)
        self.pool = nn.MaxPool2d(2)
        self.nl = nonlinearity()

    def forward(self, x):
        # skip connection
        x = x + self.bn(self.conv(x))
        x = self.nl(x)

        # downscale
        x = self.nl(self.pool(self.bn2(self.conv2(x))))
        return x


class _MaxPoolFC(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.dummy = nn.Identity()  # provides output for the hook
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor):
        x = x.amax((-2, -1))  # global max pooling (N, C, H, W) -> (N, C)
        x = self.dummy(x)
        return self.fc(x)


class CIFARNet(MetricNetwork):
    """
    Based on the current 94% CIFAR-10 speedrun world record holder by Keller Jordan.
    https://github.com/KellerJordan/cifar10-airbench
    """

    def __init__(
        self, in_shape: torch.Size, nonlinearity: NL_TYPE, hyperparams: HPARAM_TYPE
    ):
        super().__init__(hyperparams, ["in_shape", "nonlinearity"])

        channels = (in_shape[0], 64, 128, 256)
        self.blocks = nn.Sequential(
            *(
                _Block(in_channels, out_channels, nonlinearity)
                for in_channels, out_channels in zip(channels[:-1], channels[1:])
            )
        )
        self.head = _MaxPoolFC(channels[-1], hyperparams[3])

        # update return nodes (output hooks)
        self._register_hooks(
            {f"nc_layer_{i + 1}": block for i, block in enumerate(self.blocks)}
            | {"nc_output": self.head.dummy}
        )
        self.num_blocks = len(self.blocks)

    def forward(self, x: torch.Tensor):
        return self.head(self.blocks(x))

    def get_encoder_decoder(self, encoder_blocks):
        super()._check_encoder_blocks(encoder_blocks)
        encoder = nn.Sequential(*self.blocks[:encoder_blocks])
        decoder = nn.Sequential(*self.blocks[encoder_blocks:], self.head)
        return encoder, decoder


class ConvNeXt(MetricNetwork):
    def __init__(self, in_shape: torch.Size, hyperparams: HPARAM_TYPE):
        super().__init__(hyperparams, ["in_shape"])

        # import torchvision model
        self.convnext = convnext_tiny(num_classes=hyperparams[3])

        # replace first layer to match input shape
        # (parameters copied from the original model)
        self.convnext.features[0][0] = nn.Conv2d(
            in_channels=in_shape[0], out_channels=96, kernel_size=4, stride=4
        )

        # update return nodes (output hooks)
        new_hooks = {}
        for i, j in zip(range(4), (2, 2, 8, 2)):
            new_hooks[f"nc_layer_{i}"] = self.convnext.features[2 * i + 1][j]
        new_hooks["nc_output"] = self.convnext.classifier[1]
        self._register_hooks(new_hooks)
        self.num_blocks = len(self.convnext.features) // 2

    def forward(self, x):
        return self.convnext(x)

    def get_encoder_decoder(self, encoder_blocks):
        self._check_encoder_blocks(encoder_blocks)
        encoder = self.convnext.features[: 2 * encoder_blocks]
        decoder = nn.Sequential(
            *self.convnext.features[2 * encoder_blocks :],
            self.convnext.avgpool,
            self.convnext.classifier,
        )
        return encoder, decoder


class ResNet(MetricNetwork):
    def __init__(self, in_shape: torch.Size, hyperparams: HPARAM_TYPE):
        super().__init__(hyperparams, ["in_shape"])

        # import torchvision model
        self.resnet = resnet18(num_classes=hyperparams[3])

        # replace first layer to match input shape
        # (parameters copied from the original model)
        self.resnet.conv1 = nn.Conv2d(
            in_channels=in_shape[0],
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # update return nodes (output hooks)
        new_hooks = {}
        for i in range(4):
            layer = getattr(self.resnet, f"layer{i + 1}")
            new_hooks[f"nc_layer_{i}"] = layer[1].relu
        new_hooks["nc_output"] = self.resnet.avgpool
        self._register_hooks(new_hooks)
        self.num_blocks = 4

    def forward(self, x):
        return self.resnet(x)

    def get_encoder_decoder(self, encoder_blocks):
        self._check_encoder_blocks(encoder_blocks)
        encoder = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            *[getattr(self.resnet, f"layer{i}") for i in range(1, encoder_blocks + 1)],
        )
        decoder = nn.Sequential(
            *[getattr(self.resnet, f"layer{i}") for i in range(encoder_blocks + 1, 5)],
            self.resnet.avgpool,
            self.resnet.fc,
        )
        return encoder, decoder
