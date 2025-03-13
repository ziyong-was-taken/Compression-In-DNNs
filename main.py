import os

from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger

from datasets import DataModule, DecoderData
import networks
from utils import convert_to_classes, get_args


if __name__ == "__main__":
    args = get_args()

    # convert strings to class constructors
    dataset, nonlinearity, criterion, optimiser = convert_to_classes(
        args.dataset, args.nonlinearity, args.loss, args.optimiser
    )

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
        case "ConvNeXt" | "ResNet":
            model = getattr(networks, args.model)(
                dm.input_size, dm.num_classes, hyperparams
            )

    # train model (cannot use multiple devices due to nested training)
    logger = CSVLogger(os.getcwd())
    Trainer(devices=1, max_epochs=args.epochs, logger=logger).fit(model, datamodule=dm)
