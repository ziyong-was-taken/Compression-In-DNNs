from collections import OrderedDict
from copy import deepcopy

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping
import torch
from torch import nn, optim
from torch.utils.hooks import RemovableHandle
from torchvision.models import convnext_tiny, resnet18

from datasets import DIBData

NL_TYPE = type[nn.ReLU | nn.Tanh]
LOSS_TYPE = type[nn.CrossEntropyLoss | nn.MSELoss]
OPT_TYPE = type[optim.AdamW | optim.Adam | optim.SGD]
HPARAM_TYPE = tuple[LOSS_TYPE, OPT_TYPE, int, int, DIBData]


class _Network(LightningModule):
    """
    The blueprint for a basic network with a softmax output layer.
    Simply implement `_forward` and the blueprint handles the rest.
    """

    def __init__(self, criterion: LOSS_TYPE, optimiser: OPT_TYPE):
        super().__init__()

        # store architecture parameters
        self.criterion = criterion
        self.optimiser = optimiser

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
        return self.optimiser(self.parameters())


class DIBNetwork(_Network):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        num_decoders: int,
        optimiser: OPT_TYPE,
    ):
        super().__init__(criterion=nn.CrossEntropyLoss, optimiser=optimiser)
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
        return self.optimiser(self.decoders.parameters())


class _MetricNetwork(_Network):
    """
    A blueprint which extends `_Network` by also
    computing the metrics at the end of each epoch.
    """

    def __init__(
        self,
        criterion: LOSS_TYPE,
        optimiser: OPT_TYPE,
        num_decoders: int,
        dib_epochs: int,
        dib_dm: DIBData,
        num_classes: int,
    ):
        super().__init__(criterion, optimiser)

        # store DIB network parameters
        self.num_decoders = num_decoders
        self.dib_epochs = dib_epochs
        self.dib_dm = dib_dm
        self.dib_net = None

        # store handles to hooks (not strictly necessary, but good practice)
        self.hook_handles: dict[str, RemovableHandle] = {}

        # storage for NC metrics
        # zˡ: size of flattened activations of layer l
        # layer_metrics[l]: ((num_classes + zˡ) × zˡ)
        #     layer_metrics[l][:num_classes]: class sums of activations (num_classes × zˡ)
        #     layer_metrics[l][num_classes:]: gram matrix (zˡ × zˡ)
        # TODO: convert dicts to buffers
        self.num_classes = num_classes
        self.class_counts = nn.Buffer(torch.zeros(num_classes))
        self.batch_layer_outputs: dict[str, torch.Tensor] = {}
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
                self.batch_layer_outputs[name] = output

            return hook

        # register new hooks
        new_hooks |= {"nc_output": self.softmax}
        for name, module in new_hooks.items():
            self.hook_handles[name] = module.register_forward_hook(get_hook(name))

    @torch.no_grad()
    def on_train_batch_end(self, _outputs, batch: list[torch.Tensor], _batch_idx):
        """Update NC metrics"""
        _, targets = batch

        # only update class counts on first epoch
        if self.current_epoch == 0:
            self.class_counts += targets.sum(dim=0)

        # update other metrics
        for layer, activations in self.batch_layer_outputs.items():
            activations = activations.flatten(start_dim=1)
            if layer not in self.layer_metrics:
                act_size = activations.size(1)
                self.layer_metrics[layer] = torch.zeros(
                    (self.num_classes + act_size, act_size)
                ).to(targets)
            self.layer_metrics[layer][: self.num_classes] += targets.T @ activations
            self.layer_metrics[layer][self.num_classes :] += activations.T @ activations

    def on_train_epoch_end(self):
        """
        Compute and log NC metrics.
        Also train decoders and log decodable info.
        """
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
                    between_cov, within_cov, rcond=1e-5
                ).solution.trace()
        self.log_dict(nc)

        # reset NC metrics for next epoch (class counts don't change)
        self.layer_metrics.clear()

        # initialise DIB network
        if self.dib_net is None:
            self.dib_net = DIBNetwork(  # TODO: try different block indices
                *self.get_encoder_decoder(block_idx=3),
                self.num_decoders,
                self.optimiser,
            )
            if torch.cuda.is_available():
                self.dib_net.compile()

        # train DIB network
        dib_trainer = Trainer(
            devices=10,
            max_epochs=self.dib_epochs,
            logger=False,  # don't write (but do store) training losses
            default_root_dir="lightning_logs",
            deterministic=True,
            callbacks=[EarlyStopping(monitor="train_loss")],
        )
        dib_trainer.fit(self.dib_net, datamodule=self.dib_dm)

        # log final training loss, i.e., decodable information
        v_info = dib_trainer.logged_metrics["train_loss"]
        self.log("v_info", v_info)

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
