import os

from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger
from torch import nn, optim
import torchvision.datasets as torchdata

import datasets
from datasets import DataModule, DecoderData, DATASET_TYPE
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

    # setup decoder datamodule
    decoder_dm = DecoderData(
        dataset,
        data_dir="data",
        batch_size=args.decoder_batch_size,
        base_expansion=dm.base_expansion,
        num_classes=dm.num_classes,
    )
    decoder_dm.prepare_data()
    decoder_dm.setup("fit")

    # create model
    num_decoders = dm.base_expansion.size(1)
    hyperparams = (criterion, optimiser, num_decoders, args.decoder_epochs, decoder_dm)
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

    # train model (cannot use multiple devices due to nested training)
    logger = CSVLogger(os.getcwd())
    Trainer(
        devices=1,
        max_epochs=args.epochs,
        logger=logger,
        deterministic=True,
    ).fit(model, datamodule=dm)
