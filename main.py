import os

import torchvision.datasets as torchdata
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.tuner.tuning import Tuner
from torch import nn, optim

import datasets
import networks
from datasets import DATASET_TYPE, DataModule, DIBData
from networks import LOSS_TYPE, NL_TYPE, OPT_TYPE
from utils import ComputeDIB, ComputeNC1, base_expand, get_args

args = get_args()
seed_everything(seed=0, workers=True)

# convert strings to class constructors
dataset: DATASET_TYPE
if hasattr(torchdata, args.dataset):
    dataset = getattr(torchdata, args.dataset)
else:  # fallback to custom dataset
    dataset = getattr(datasets, args.dataset)
nonlinearity: NL_TYPE = getattr(nn, args.nonlinearity)
criterion: LOSS_TYPE = getattr(nn, args.loss + "Loss")
optimiser: OPT_TYPE = getattr(optim, args.optimiser)

# setup main datamodule
dm = DataModule(dataset, data_dir=args.data_dir, batch_size=args.batch_size)
dm.prepare_data()
dm.setup("fit")

# setup DIB datamodule
new_labels = base_expand(dm.labels, dm.num_classes)
dib_dm = DIBData(dm, new_labels)
dib_dm.prepare_data()
dib_dm.setup("fit")

# create model
hyperparams = (criterion, optimiser, args.learning_rate)
match args.model:
    case "MLP":
        model = networks.MLP(
            [dm.input_size.numel()] + args.widths + [dm.num_classes],
            nonlinearity,
            hyperparams,
        )
    case _:
        model = getattr(networks, args.model)(
            dm.input_size, dm.num_classes, hyperparams
        )
model.compile(disable=not args.compile)

# # create tuner
# dummy_trainer = Trainer(devices=1, max_epochs=-1, barebones=True, deterministic=True)
# tuner = Tuner(dummy_trainer)

# # tune batch size
# tuner.scale_batch_size(model, datamodule=dm, batch_arg_name="batch_size")
# dib_dm.batch_size = dm.batch_size // args.num_devices

# # tune learning rate (broken?)
# lr_finder = tuner.lr_find(
#     model, datamodule=dm, update_attr=True, attr_name="learning_rate"
# )
# lr_finder.plot(suggest=True, show=True)
# print("Learning rate:", model.learning_rate)

# train model
logger = CSVLogger(os.getcwd())
Trainer(
    devices=1,  # only one device due to nested training
    max_epochs=args.epochs,
    logger=logger,
    deterministic=True,
    callbacks=[
        ComputeDIB(
            num_decoders=new_labels.size(1),
            dib_epochs=args.dib_epochs,
            dib_dm=dib_dm,
            num_devices=args.num_devices,
            block_indices=list(range(model.num_blocks)),
            no_compile=not args.compile,
        ),
        ComputeNC1(dm.num_classes),
        EarlyStopping(monitor="train_loss", patience=30),
    ],
).fit(model, datamodule=dm)
