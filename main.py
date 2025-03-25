import os

from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.tuner import Tuner
from torch import nn, optim, cuda
import torchvision.datasets as torchdata

import datasets
from datasets import DataModule, DIBData, DATASET_TYPE
import networks
from networks import NL_TYPE, LOSS_TYPE, OPT_TYPE
from utils import get_args


if __name__ == "__main__":
    args = get_args()
    seed_everything(seed=0, workers=True)

    # convert strings to class constructors
    dataset: DATASET_TYPE = getattr(  # fallback to custom datasets
        torchdata, args.dataset, getattr(datasets, args.dataset)
    )
    nonlinearity: NL_TYPE = getattr(nn, args.nonlinearity)
    criterion: LOSS_TYPE = getattr(nn, args.loss + "Loss")
    optimiser: OPT_TYPE = getattr(optim, args.optimiser)

    # setup main datamodule
    dm = DataModule(dataset, data_dir="data")
    dm.prepare_data()
    dm.setup("fit")

    # setup DIB datamodule
    dib_dm = DIBData(dm)
    dib_dm.prepare_data()
    dib_dm.setup("fit")

    # create model
    num_decoders = dm.base_expansion.size(1)
    hyperparams = (criterion, optimiser, num_decoders, args.dib_epochs, dib_dm)
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
    if cuda.is_available():
        model.compile()

    # tune batch size
    dummy_trainer = Trainer(
        devices=1, max_epochs=-1, barebones=True, deterministic=True
    )
    tuner = Tuner(dummy_trainer)
    tuner.scale_batch_size(model, datamodule=dm, batch_arg_name="batch_size")
    dib_dm.batch_size = dm.batch_size // args.num_devices

    # # tune learning rate (broken for now)
    # lr_finder = tuner.lr_find(
    #     model, datamodule=dm, update_attr=True, attr_name="learning_rate"
    # )
    # lr_finder.plot(suggest=True, show=True)
    # print("Learning rate:", model.learning_rate)

    # train model
    # logger = CSVLogger(os.getcwd())
    # Trainer(
    #     devices=1,  # only one device due to nested training
    #     max_epochs=args.epochs,
    #     logger=logger,
    #     deterministic=True,
    # ).fit(model, datamodule=dm)
