import argparse
import os

from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger

from datasets import DataModule, DecoderData
import networks
from utils import convert_to_classes


def get_args():
    """
    Parse command line arguments:

    -m, --model: model to use
    -w, --widths: widths of hidden layers of MLP
    -nl, --nonlinearity: nonlinearity used in hidden layers of MLP
    --dataset: dataset to use
    -opt, --optimiser: optimiser to use
    --loss: loss function to use
    --epochs: number of epochs to train the model
    -bs, --batch-size: batch size
    --decoder-epochs: number of epochs to train the decoder
    --decoder-batch-size: batch size for the decoder
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        default="MLP",
        choices=["MLP", "ConvNeXt", "ResNet"],
        help="model to use",
    )
    parser.add_argument(
        "-w",
        "--widths",
        nargs="+",
        default=[12, 10, 7, 5, 4, 3, 2],  # same model as ib-2017
        type=int,
        help="widths of hidden layers of MLP, has no effect on ConvNeXt and ResNet (yet)",
        metavar="WIDTH",
    )
    parser.add_argument(
        "-nl",
        "--nonlinearity",
        default="ReLU",
        help=(
            "nonlinearity used in hidden layers of MLP, "
            "has no effect on ConvNeXt and ResNet (yet), "
            "case-sensitive"
        ),
    )
    parser.add_argument(
        "--dataset",
        default="MNIST",
        choices=["MNIST", "CIFAR10", "FashionMNIST"],
        help="dataset to use, case-sensitive",
    )
    parser.add_argument(
        "-opt",
        "--optimiser",
        default="AdamW",
        choices=["AdamW", "Adam", "SGD"],
        help="optimiser to use, case-sensitive",
    )
    parser.add_argument(
        "--loss",
        default="CrossEntropy",
        choices=["CrossEntropy", "MSE"],
        help="loss function to use, case-sensitive",
    )
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("-bs", "--batch-size", default=64, type=int)
    parser.add_argument("--decoder-epochs", default=50, type=int)
    parser.add_argument("--decoder-batch-size", default=64, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # convert strings to functions
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

    # setup trainers
    logger = CSVLogger(os.getcwd())
    decoder_trainer = Trainer(devices=1, max_epochs=args.decoder_epochs, logger=logger)
    trainer = Trainer(devices=10, max_epochs=args.epochs, logger=logger)

    # create model
    hyperparams = (criterion, optimiser, decoder_trainer, decoder_dm)
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

    # train model
    trainer.fit(model, datamodule=dm)
