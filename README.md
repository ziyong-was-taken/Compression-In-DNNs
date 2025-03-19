# Explaining Deep Neural Networks using Information Theory and Geometry

This repository contains the code to reproduce the results of my eponymously named master's thesis.
It is mainly implemented using [(PyTorch) Lightning](https://lightning.ai/docs/pytorch/stable/).

<!-- omit in toc -->
## Table of Contents

- [Usage](#usage)
  - [Requirements](#requirements)
  - [Flags](#flags)
- [Documentation](#documentation)
- [Implementation Details](#implementation-details)
  - [Modified Algorithm 1](#modified-algorithm-1)
    - [Example](#example)
  - [NC1 Computation](#nc1-computation)
  - [DIB Computation](#dib-computation)
  - [Rank](#rank)

## Usage

The easiest way to run the code is use [conda](https://docs.conda.io/en/latest/).
Simply create a new environment using the provided `environment.yaml` file using the following commands:

```bash
# create and activate the environment (name: master_thesis)
conda env create -f environment.yaml
conda activate master_thesis

# run the code (possibly with flags)
python main.py
```

### Requirements

- `lightning>=2.5.0`: [(PyTorch) Lightning](https://lightning.ai/docs/pytorch/stable/) is the main framework used
- `torchvision>=0.21.0`: datasets and models are imported from [torchvision](https://pytorch.org/vision/stable/index.html)
- `ipykernel>=6.29.5`: necessary to run Jupyter notebook
- `matplotlib>=3.10.1`: used to create plots
- `pandas>=2.2.3`: used to read csv data

### Flags

- flags include `-m` to choose the model and `-d` to choose the dataset
- for example,
  <!---->
  ```bash
  python main.py -m MLP -d MNIST
  ```
  <!---->
  uses a multi-layer perceptron model and the MNIST dataset
- supported models (case-sensitive):
  - `MLP`: multi-layer perceptron with custom widths specified by `-w` and nonlinearity specified by `-nl`
  - `ConvNeXt`: the [ConvNeXt-T architecture](https://pytorch.org/vision/0.21/models/generated/torchvision.models.convnext_tiny.html) from [A ConvNet for the 2020s](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html)
  - `ResNet`: the [ResNet-18 architecture](https://pytorch.org/vision/0.21/models/generated/torchvision.models.resnet18.html) from [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
- supported datasets (case-sensitive):
  - `MNIST`: [MNIST](http://yann.lecun.com/exdb/mnist/)
  - `CIFAR10`: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
  - `FashionMNIST`: [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
- for a full list of flags, run `python main.py --help`
- further information is also available in the `get_args()` function of `utils.py`

## Documentation

The code consists of four main Python modules and one Jupyter notebook:

- `utils.py`: contains the command line flag parser and a slightly modified version of Algorithm 1 of "Learning Optimal Representations with the Decodable Information Bottleneck" (see [Modified Algorithm 1](#modified-algorithm-1))
- `main.py`: "glue code" which sets up the dataset(s), then creates and trains the model
- `datasets.py`: logic for loading and transforming the dataset(s)
- `networks.py`: the meat of the project; see [Implementation Details](#implementation-details) for more details
- `plots.ipynb`: notebook for creating plots using data

When using a dataset for the first time, Lightning will download it into `data/`.
During training run $i$, Lightning stores model checkpoints in `lightning_logs/version_`$2i$`/checkpoints/`
(as well as duplicate checkpoints in `lightning_logs/version_`$2i+1$`/checkpoints/` due to a bug when training nested models).
The metrics for each epoch are stored in `lightning_logs/version_`$2i$`/metrics.csv`
and the hyperparameters in `lightning_logs/version_`$2i$`/hparams.yaml`.

## Implementation Details

Metrics are computed after each epoch using the `on_train_epoch_end` callback.

### Modified Algorithm 1

1. for each class $y$, enumerate the samples $\mathcal X_y = \{x \mid x$ has label $y\}$, i.e., assign them the indices $0,1,…,|\mathcal X_y| - 1$
2. convert each index to base $C$ (the number of classes), implicitly padding with zeros to the left
3. the new labels for each sample are the digits of its base-$C$ representation

#### Example

Let $C = 5$ and the samples be labeled [0, 0, 0, 1, 1, 3, 4, 4, 4, 4, 4, 4].
Then, the samples are assigned the indices [0, 1, 2, 0, 1, 0, 0, 1, 2, 3, 4, 5].
Converting to base $C = 5$ and zero-padding to the left gives \[[0, 0], [0, 1], [0, 2], [0, 0], [0, 1], [0, 0], [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 0]\].
The new labels are then [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] and [0, 1, 2, 0, 1, 0, 0, 1, 2, 3, 4, 0].

### NC1 Computation

- goal: compute $\operatorname{tr}(Σ_W^l Σ_B^{l+})$ for all layers $l ∈ ℒ$
- the activations $\{𝐡ˡ_{c,i}\}_{l∈ℒ,\ c ∈ \{1,…,C\},\ i ∈ \{1,…,N\}}$ are accessed by registering forward hooks:
  - MLP: at the end of each nonlinearity
  - ConvNeXt-T, ResNet-18: at the end of each residual block
  - these hooks store the output of each hooked layer after each forward pass
- issue: activations don't fit in memory all at once
- solution: only store batch activations $\{𝐡ˡ_{c,i}\}_{l ∈ ℒ,\ c ∈ \{1,…,C\},\ i ∈ \{1,…,S\}}$ where $S$ is the batch size
- after training a batch and triggering all forward hooks, the `on_train_batch_end` callback is used to update the running
  - class counts $\{n_c\}_{c=1}^C$
  - class totals $\{\{∑_{i=1}^{n_c} 𝐡_{c,i}^l\}_{c=1}^C\}_{l ∈ ℒ}$
  - gram matrices $G^l = ∑_{c=1}^C ∑_{i=1}^{n_c} 𝐡_{c,i}^l 𝐡_{c,i}^{l⊤}$ (useful later)
- computing $Σ_B^l$
  - $\{\{\boldsymbol μ_c^l\}_{c=1}^C\}_{l ∈ ℒ}$: straightforward using class counts and class totals
  - $\{\bar{\boldsymbol μ}^l\}_{l ∈ ℒ}$: simply sum class counts and class totals
  - $Σ_B^l = 1/C ∑_{c=1}^C ({\boldsymbol μ}_c^l - \bar{\boldsymbol μ}^l)({\boldsymbol μ}_c^l - \bar{\boldsymbol μ}^l)^⊤$
- computing $Σ_W^l$
  - recall: $Σ_W^l + Σ_B^l = Σ_T^l = 1/N ∑_{c=1}^C ∑_{i=1}^{n_c}(𝐡_{c,i}^l - \bar{\boldsymbol μ}^l)(𝐡_{c,i}^l - \bar{\boldsymbol μ}^l)^⊤$
  - lemma: $Σ_T^l = G^l/N - \bar{\boldsymbol μ}^l \bar{\boldsymbol μ}^{l⊤}$
    $$
    \begin{align*}
      Σ_T^l
        &= 1/N ∑_{c=1}^C ∑_{i=1}^{n_c}(𝐡_{c,i}^l - \bar{\boldsymbol μ}^l)(𝐡_{c,i}^l - \bar{\boldsymbol μ}^l)^⊤ \\
        &= 1/N ∑_{c=1}^C ∑_{i=1}^{n_c} (𝐡_{c,i}^l 𝐡_{c,i}^{l⊤} - \bar{\boldsymbol μ}^l 𝐡_{c,i}^{l⊤} - 𝐡_{c,i}^l \bar{\boldsymbol μ}^{l⊤} + \bar{\boldsymbol μ}^l \bar{\boldsymbol μ}^{l⊤}) \\
        &= 1/N ∑_{c=1}^C ∑_{i=1}^{n_c} 𝐡_{c,i}^l 𝐡_{c,i}^{l⊤} - 1/N ∑_{c=1}^C ∑_{i=1}^{n_c} (\bar{\boldsymbol μ}^l 𝐡_{c,i}^{l⊤} + 𝐡_{c,i}^l \bar{\boldsymbol μ}^{l⊤}) + \bar{\boldsymbol μ}^l \bar{\boldsymbol μ}^{l⊤} \\
        &= G^l/N - \bar{\boldsymbol μ}^l\left(1/N ∑_{c=1}^C ∑_{i=1}^{n_c} 𝐡_{c,i}^{l⊤}\right) - \left(1/N ∑_{c=1}^C ∑_{i=1}^{n_c} 𝐡_{c,i}^l\right) \bar{\boldsymbol μ}^{l⊤} + \bar{\boldsymbol μ}^l \bar{\boldsymbol μ}^{l⊤} \\
        &= G^l/N - \bar{\boldsymbol μ}^l \bar{\boldsymbol μ}^{l⊤} - \bar{\boldsymbol μ}^l \bar{\boldsymbol μ}^{l⊤} + \bar{\boldsymbol μ}^l \bar{\boldsymbol μ}^{l⊤} \\
        &= G^l/N - \bar{\boldsymbol μ}^l \bar{\boldsymbol μ}^{l⊤}
    \end{align*}
    $$
  - thus, $Σ_W^l = G^l/N - \bar{\boldsymbol μ}^l \bar{\boldsymbol μ}^{l⊤} - Σ_B^l$
- finally, since $\operatorname{tr}(Σ_W^l Σ_B^{l+}) = \operatorname{tr}(Σ_B^{l+} Σ_W^l)$, use `torch.linalg.lstsq` to compute $Σ_B^{l+} Σ_W^l$

### DIB Computation

- compute new labels for all samples using modified Algorithm 1
- for each new label, create a copy of the decoder $D$ returned by `model.get_encoder_decoder()`
- combine the encoder $E$ and decoders $D_1, D_2, \dots$ into a single model $M$
  <!---->
  ```markdown
         E
  M =  / | \
      D₁ D₂ …
  ```
  <!---->
- train $M$ using average cross-entropy loss over the decoder heads
- DIB is final training loss of $M$

### Rank

- TODO
