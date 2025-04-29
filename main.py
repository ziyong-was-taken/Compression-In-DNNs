import os

from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.tuner.tuning import Tuner
from torch import nn, optim
from torchvision import datasets as torchdata

import datasets
import networks
from datasets import DATASET_TYPE, DataModule, DIBData
from networks import LOSS_TYPE, NL_TYPE, OPT_TYPE, MetricNetwork
from utils import ComputeDIB, ComputeNC1, base_expand, get_args, total_steps

args = get_args()
seed_everything(seed=0)

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
dm = DataModule(
    dataset,
    data_dir=args.data_dir,
    batch_size=args.batch_size,
    num_devices=args.num_devices,
)
dm.prepare_data()
dm.setup("fit")

# setup DIB datamodule
dib_labels = base_expand(dm.labels, dm.num_classes)
dib_dm = DIBData(dm, dib_labels)
dib_dm.prepare_data()
dib_dm.setup("fit")

# create model
model: MetricNetwork
hyperparams = (
    criterion,
    optimiser,
    args.learning_rate,
    total_steps(len(dm.train), dm.batch_size, args.epochs),
)
match args.model:
    case "MLP":
        model = networks.MLP(
            [dm.input_size.numel()] + args.widths, nonlinearity, hyperparams
        )
    case "CNN":
        model = networks.CNN(dm.input_size, nonlinearity, hyperparams)
    case _:
        model = getattr(networks, args.model)(dm.input_size, dm.num_classes, hyperparams)
model.compile(disable=not args.compile, fullgraph=True, options={"max_autotune": True})

######################### hyperparameter tuning (WIP) #########################

# # create tuner
# dummy_trainer = Trainer(
#     devices=1,
#     max_epochs=-1,
#     barebones=True,
#     deterministic=True,
# )
# tuner = Tuner(dummy_trainer)

# # tune batch size
# tuner.scale_batch_size(model, datamodule=dm, batch_arg_name="batch_size")

# # tune learning rate
# lr_finder = tuner.lr_find(
#     model, datamodule=dm, update_attr=True, attr_name="learning_rate"
# )
# lr_finder.plot(suggest=True, show=True)

# print("Optimal learning rate:", model.learning_rate)

###############################################################################

# train model
logger = CSVLogger(os.getcwd())
Trainer(
    devices=args.num_devices,
    max_epochs=args.epochs,
    logger=logger,
    log_every_n_steps=1,
    benchmark=True,
    # deterministic=True, # ignored when benchmark=True
    callbacks=[
        ComputeDIB(
            num_decoders=dib_labels.size(1),
            dib_epochs=args.dib_epochs,
            dib_dm=dib_dm,
            num_devices=args.num_devices,
            block_indices=list(range(model.num_blocks)),
            no_compile=not args.compile,
        ),
        ComputeNC1(class_counts=dm.class_counts),
    ],
).fit(model, datamodule=dm)
