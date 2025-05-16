import os

from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.tuner.tuning import Tuner
from torch import nn, optim, set_float32_matmul_precision
from torchvision import datasets as torchdata

import datasets
import networks
from datasets import DATASET_TYPE, DataModule, DIBData
from networks import LOSS_TYPE, NL_TYPE, OPT_TYPE, MetricNetwork
from utils import ComputeDIB, ComputeNC1, get_args

args = get_args()
seed_everything(seed=args.seed)
set_float32_matmul_precision("high")

# convert strings to class constructors
dataset: DATASET_TYPE
if hasattr(torchdata, args.dataset):
    dataset = getattr(torchdata, args.dataset)
else:  # fallback to custom dataset
    dataset = getattr(datasets, args.dataset)
nonlinearity: NL_TYPE = getattr(nn, args.nonlinearity)
criterion: LOSS_TYPE = getattr(nn, args.loss + "Loss")
optimiser: OPT_TYPE = getattr(optim, args.optimiser)

# setup and compute metrics of dataset
dm = DataModule(dataset, args.data_dir, args.batch_size, args.num_devices)
dm.prepare_data()
dm.setup("fit")

# create trainer
logger = CSVLogger(os.getcwd())
trainer = Trainer(
    devices=args.num_devices,
    precision="bf16-true",
    max_epochs=args.epochs,
    logger=logger,
    benchmark=True,
    # deterministic=True, # ignored when benchmark=True
    num_sanity_val_steps=0,
    callbacks=[
        ComputeDIB(
            dib_dm=DIBData(dm), dib_epochs=args.dib_epochs, num_devices=args.num_devices
        ),
        ComputeNC1(dm.train_class_counts, dm.val_class_counts, dm.num_classes),
    ],
)

hyperparams = (
    criterion,
    optimiser,
    args.learning_rate,
    dm.num_classes,
    not args.compile,
)

# create model directly on the accelerator
model: MetricNetwork
with trainer.init_module():
    match args.model:
        case "MLP":
            model = networks.MLP(
                [dm.input_size.numel()] + args.widths, nonlinearity, hyperparams
            )
        case "MNISTNet" | "CIFARNet":
            model = getattr(networks, args.model)(
                dm.input_size, nonlinearity, hyperparams
            )
        case _:
            model = getattr(networks, args.model)(dm.input_size, hyperparams)

if args.tune:
    # create tuner
    dummy_trainer = Trainer(
        devices=1,
        precision="bf16-true",
        max_epochs=-1,
        barebones=True,
        benchmark=True,
        # deterministic=True, # ignored when benchmark=True
    )
    tuner = Tuner(dummy_trainer)

    # tune learning rate
    lr_finder = tuner.lr_find(model, datamodule=dm, update_attr=True)
    if not args.compile:  # plotting breaks the computation graph
        lr_finder.plot(suggest=True, show=True)

# train model
trainer.fit(model, datamodule=dm)
