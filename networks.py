from collections import OrderedDict
from copy import deepcopy

import torch
from lightning import LightningModule
from torch import nn, optim
from torchvision.models import convnext_tiny, resnet18


NL_TYPE = type[nn.ReLU | nn.Tanh]
LOSS_TYPE = type[nn.CrossEntropyLoss | nn.MSELoss]
OPT_TYPE = type[optim.AdamW | optim.Adam | optim.SGD]
HPARAM_TYPE = tuple[LOSS_TYPE, OPT_TYPE, float]


class _Network(LightningModule):
    """
    The blueprint for a basic network with a softmax output layer.
    Simply implement `_forward` and the blueprint handles the rest.
    """

    def __init__(self, criterion: LOSS_TYPE, optimiser: OPT_TYPE, learning_rate: float):
        super().__init__()

        # store architecture parameters
        self.criterion = criterion
        self.optimiser = optimiser
        self.learning_rate = learning_rate

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
        # loss = self.criterion()(self(inputs), targets) # see https://discuss.pytorch.org/t/pytorchs-non-deterministic-cross-entropy-loss-and-the-problem-of-reproducibility/172180/9
        loss = self.criterion(reduction="none")(self(inputs), targets).mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return self.optimiser(self.parameters(), lr=self.learning_rate)


class DIBNetwork(_Network):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        num_decoders: int,
        optimiser: OPT_TYPE,
        learning_rate: float,
    ):
        super().__init__(
            criterion=nn.CrossEntropyLoss,
            optimiser=optimiser,
            learning_rate=learning_rate,
        )
        self.encoder = deepcopy(encoder)  # not copying encoder messes with momentum
        self.decoders = nn.ModuleList(deepcopy(decoder) for _ in range(num_decoders))

        # freeze encoder
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

    def on_fit_start(self):
        """
        Reset decoder weights before training.
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

    def _forward(self, x):
        encoded = self.encoder(x)
        if torch.cuda.is_available():
            outputs = nn.parallel.parallel_apply(
                (*self.decoders,), (encoded,) * len(self.decoders)
            )
        else:
            outputs = [d(encoded) for d in self.decoders]
        return torch.stack(outputs, dim=-1)

    def configure_optimizers(self):
        return self.optimiser(self.decoders.parameters(), lr=self.learning_rate)


class MetricNetwork(_Network):
    """
    A blueprint which extends `_Network` with helper functions to compute NC1 and DIB
    """

    def __init__(self, hyperparams: HPARAM_TYPE):
        super().__init__(*hyperparams)
        self.num_blocks: int
        self.batch_activations: dict[str, torch.Tensor] = {}

    def _register_hooks(self, new_hooks: dict[str, nn.Module]):
        """Register forward hooks for NC metrics"""

        # get_hook: layer_name -> (hook: module, args, output -> None)
        def get_hook(name):
            def hook(_module, _args, output):
                self.batch_activations[name] = output.detach()

            return hook

        # register new hooks
        new_hooks |= {"nc_output": self.softmax}
        for name, module in new_hooks.items():
            module.register_forward_hook(get_hook(name))

    def _check_block_idx(self, block_idx):
        assert block_idx < self.num_blocks, (
            f"{type(self).__name__} only has {self.num_blocks} blocks"
        )

    def get_encoder_decoder(self, block_idx) -> tuple[nn.Module, nn.Module]:
        """
        Get a copy of the encoder up to start of block `block_idx`
        and the decoder (sans softmax) containing the rest of the network.
        """
        raise NotImplementedError


class MLP(MetricNetwork):
    def __init__(
        self, widths: list[int], nonlinearity: NL_TYPE, hyperparams: HPARAM_TYPE
    ):
        super().__init__(hyperparams)

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
        self.fc = nn.Linear(widths[-2], widths[-1])

        # update return nodes (output hooks)
        super()._register_hooks(
            {f"nc_layer_{i}": block.nl for i, block in enumerate(self.blocks)}
        )
        self.num_blocks = len(self.blocks)

    def _forward(self, x):
        return self.fc(self.blocks(self.flatten(x)))

    def get_encoder_decoder(self, block_idx):
        super()._check_block_idx(block_idx)
        encoder = nn.Sequential(self.flatten, *self.blocks[:block_idx])
        decoder = nn.Sequential(*self.blocks[block_idx:], self.fc)
        return encoder, decoder


class ConvNeXt(MetricNetwork):
    def __init__(
        self, in_shape: torch.Size, num_classes: int, hyperparams: HPARAM_TYPE
    ):
        super().__init__(hyperparams)

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
        self.num_blocks = len(self.convnext.features) // 2

    def _forward(self, x):
        return self.convnext(x)

    def get_encoder_decoder(self, block_idx):
        super()._check_block_idx(block_idx)
        encoder = self.convnext.features[: 2 * block_idx]
        decoder = nn.Sequential(
            *self.convnext.features[2 * block_idx :],
            self.convnext.avgpool,
            self.convnext.classifier,
        )
        return encoder, decoder


class ResNet(MetricNetwork):
    def __init__(
        self, in_shape: torch.Size, num_classes: int, hyperparams: HPARAM_TYPE
    ):
        super().__init__(hyperparams)

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
        self.num_blocks = 4

    def _forward(self, x):
        return self.resnet(x)

    def get_encoder_decoder(self, block_idx):
        super()._check_block_idx(block_idx)
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
