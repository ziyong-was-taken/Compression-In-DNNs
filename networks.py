from collections import OrderedDict
from copy import deepcopy

import torch
from lightning import LightningModule
from torch import nn, optim
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional.classification import accuracy
from torchvision.models import convnext_tiny, resnet18

NL_TYPE = type[nn.ReLU | nn.Tanh]
LOSS_TYPE = type[nn.CrossEntropyLoss]
OPT_TYPE = type[optim.AdamW | optim.Adam | optim.SGD]
HPARAM_TYPE = tuple[
    LOSS_TYPE, OPT_TYPE, float, int, int
]  # loss function, optimiser, learning rate, number of classes, total steps


class _Network(LightningModule):
    """
    The blueprint for a basic network with a softmax output layer.
    Simply implement `forward` and the blueprint handles the rest.
    """

    def __init__(
        self,
        criterion: LOSS_TYPE,
        optimiser: OPT_TYPE,
        learning_rate: float,
        num_classes: int,
        total_steps: int,
    ):
        super().__init__()

        # store architecture parameters
        self.criterion = criterion
        self.optimiser = optimiser
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.total_steps = total_steps

    def forward(self, x):
        raise NotImplementedError

    def training_step(self, batch: list[torch.Tensor]):
        """Compute and log average training loss"""
        inputs, targets = batch
        preds: torch.Tensor = self(inputs)
        loss = self.criterion()(preds, targets)
        self.log("train_loss", loss, sync_dist=True)

        with torch.inference_mode():
            train_acc = accuracy(
                preds.argmax(dim=1), targets, "multiclass", num_classes=self.num_classes
            )
            self.log("train_acc", train_acc, sync_dist=True)
            # self.log("lr", self.lr_schedulers().get_last_lr()[0])

        return loss

    def _opt_parameters(self):
        """Parameters to optimise"""
        return self.parameters()

    def configure_optimizers(self):
        optimiser = self.optimiser(
            self._opt_parameters(), lr=self.learning_rate, fused=True
        )
        lr_scheduler = OneCycleLR(
            optimiser,
            max_lr=self.learning_rate,
            total_steps=self.total_steps,
            cycle_momentum=False,
        )
        return {
            "optimizer": optimiser,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }


class DIBNetwork(_Network):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        num_decoders: int,
        optimiser: OPT_TYPE,
        learning_rate: float,
        num_classes: int,
        total_steps: int,
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
        super().__init__(
            criterion=nn.CrossEntropyLoss,
            optimiser=optimiser,
            learning_rate=learning_rate,
            num_classes=num_classes,
            total_steps=total_steps,
        )
        # copy encoder to ensure correct device placement of parameters
        self.encoder = deepcopy(encoder).requires_grad_(False).eval()
        self.decoders = nn.ModuleList(deepcopy(decoder) for _ in range(num_decoders))

    def update_encoder(self, encoder: nn.Module):
        """Update the parameters of the encoder"""
        self.encoder.load_state_dict(encoder.state_dict())

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

    def forward(self, x):
        encoded = self.encoder(x)
        if torch.cuda.is_available():
            outputs = nn.parallel.parallel_apply(
                (*self.decoders,), (encoded,) * len(self.decoders)
            )
        else:
            outputs = [d(encoded) for d in self.decoders]
        return torch.stack(outputs, dim=-1)

    def _opt_parameters(self):
        return self.decoders.parameters()


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
        for name, module in new_hooks.items():
            module.register_forward_hook(get_hook(name))

    # TODO
    def validation_step(self, batch: list[torch.Tensor]):
        pass

    # TODO
    def test_step(self, batch: list[torch.Tensor]):
        pass

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
        super().__init__(hyperparams)

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
            {f"nc_layer_{i}": block.nl for i, block in enumerate(self.blocks)}
            | {"nc_output": self.fc}
        )
        self.num_blocks = len(self.blocks)

    def forward(self, x):
        return self.fc(self.blocks(self.flatten(x)))

    def get_encoder_decoder(self, encoder_blocks):
        self._check_encoder_blocks(encoder_blocks)
        encoder = nn.Sequential(self.flatten, *self.blocks[:encoder_blocks])
        decoder = nn.Sequential(*self.blocks[encoder_blocks:], self.fc)
        return encoder, decoder


class CNN(MetricNetwork):
    """
    Tuomas Oikarinen's performance-focussed (both in speed and accuracy) CNN.
    https://github.com/tuomaso/train_mnist_fast
    """

    def __init__(
        self,
        in_shape: torch.Size,
        nonlinearity: NL_TYPE,
        hyperparams: HPARAM_TYPE,
    ):
        super().__init__(hyperparams)

        # define convolutional blocks
        channels = (in_shape[0], 24, 32)
        kernels = (5, 3)
        assert len(channels) == len(kernels) + 1, (
            f"Number of channels ({len(channels)}) ≠ number of kernels ({len(kernels)}) + 1"
        )
        self.blocks = nn.Sequential(
            *[
                nn.Sequential(
                    OrderedDict(
                        {
                            "conv": nn.Conv2d(*conv_params),
                            "pool": nn.MaxPool2d(2),
                            "nl": nonlinearity(),
                        }
                    )
                )
                for conv_params in zip(channels[:-1], channels[1:], kernels)
            ]
        )

        def conv_final_ndim():
            """Helper function to compute the dimension after the convolutional layers"""
            res = channels[-1]
            for dim in in_shape[1:]:
                for k in kernels:
                    dim = (dim - (k - 1)) // 2
                res *= dim
            return res

        # define fully connected layers
        self.classifier = nn.Sequential(
            OrderedDict(
                {
                    "flatten": nn.Flatten(),
                    "fc1": nn.Linear(conv_final_ndim(), 256),
                    "nl": nonlinearity(),
                    "fc2": nn.Linear(256, hyperparams[3]),
                }
            )
        )

        # update return nodes (output hooks)
        self._register_hooks(
            {f"nc_layer_{i}": block.nl for i, block in enumerate(self.blocks)}
            | {"nc_output": self.classifier.fc2}
        )
        self.num_blocks = len(self.blocks)

    def forward(self, x):
        return self.classifier(self.blocks(x))

    def get_encoder_decoder(self, encoder_blocks):
        super()._check_encoder_blocks(encoder_blocks)
        encoder = nn.Sequential(*self.blocks[:encoder_blocks])
        decoder = nn.Sequential(*self.blocks[encoder_blocks:], self.classifier)
        return encoder, decoder


class ConvNeXt(MetricNetwork):
    def __init__(self, in_shape: torch.Size, hyperparams: HPARAM_TYPE):
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
            self.convnext.classifier[2].in_features, hyperparams[3]
        )

        # update return nodes (output hooks)
        new_hooks = {}
        for i, j in zip(range(4), (2, 2, 8, 2)):
            new_hooks[f"nc_layer_{i}"] = self.convnext.features[2 * i + 1][j]
        new_hooks["nc_output"] = self.convnext.classifier[2]
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
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, hyperparams[3])

        # update return nodes (output hooks)
        new_hooks = {}
        for i in range(4):
            layer = getattr(self.resnet, f"layer{i + 1}")
            new_hooks[f"nc_layer_{i}"] = layer[1].relu
        new_hooks["nc_output"] = self.resnet.fc
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
