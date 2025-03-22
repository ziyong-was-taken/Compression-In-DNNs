import os

from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger
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
    dm = DataModule(dataset, data_dir="data", batch_size=args.batch_size)
    dm.prepare_data()
    dm.setup("fit")

    # setup DIB datamodule
    dib_dm = DIBData(
        dataset,
        data_dir="data",
        batch_size=args.dib_batch_size,
        base_expansion=dm.base_expansion,
        num_classes=dm.num_classes,
    )
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

    # create trainer (cannot use multiple devices due to nested training)
    logger = CSVLogger(os.getcwd())
    trainer = Trainer(
        devices=1,
        max_epochs=args.epochs,
        logger=logger,
        deterministic=True,
    )

    # TODO: tune hyperparameters (batch size, learning rate)
    
    # train model
    trainer.fit(model, datamodule=dm)
