# Explaining Deep Neural Networks using Information Theory and Geometry

This repository contains the code to reproduce the results of my eponymously named master's thesis.
It is mainly implemented using [(PyTorch) Lightning](https://lightning.ai/docs/pytorch/stable/).

<!-- omit in toc -->
## Table of Contents

- [Usage](#usage)
  - [Requirements](#requirements)
  - [Flags](#flags)
  - [Plotting Results](#plotting-results)
- [Code Structure](#code-structure)
- [Algorithms](#algorithms)
  - [Modified Algorithm 1](#modified-algorithm-1)
  - [NC1 Computation](#nc1-computation)
  - [DIB Computation](#dib-computation)

## Usage

The easiest way to run the code is to use [conda](https://conda.org/).
Simply create a new environment using the provided `environment.yaml` file and the following commands:

```bash
# create and activate the environment (name: master_thesis)
conda env create -f environment.yaml
conda activate master_thesis

# run the code (possibly with flags)
python main.py --flag value

# (optional) deactivate the environment afterwards
conda deactivate
```

The repository also includes a minimal [micromamba-docker](https://micromamba-docker.readthedocs.io/en/latest/index.html) Dockerfile.
Build and run the [Docker](https://www.docker.com/) image by using the following commands:

```bash
# build the Docker image (tag: master_thesis)
docker build -t master_thesis .

# run the Docker container
docker run -it master_thesis python main.py --flag value

# run the Docker container (with GPUs)
docker run -it --gpus all master_thesis python main.py --flag value
```

There is also an [Apptainer](https://apptainer.org/) definition file `master_thesis.def` which can be used like so:

```bash
# build the Apptainer container (filename: master_thesis.sif)
apptainer build master_thesis.sif master_thesis.def

# run the Apptainer container
./master_thesis.sif python main.py --flag value

# run the Apptainer container (with GPUs)
apptainer run --nv master_thesis.sif python main.py --flag value
```

### Requirements

- `lightning>=2.5.1`: [(PyTorch) Lightning](https://lightning.ai/docs/pytorch/stable/) is the main framework used
- `torchvision>=0.21.0`: datasets and models are imported from [torchvision](https://pytorch.org/vision/stable/index.html)
  - note: The conda solver prioritises the CPU build of `torchvision` (over the CUDA build) due to its higher build number.
  - To force the CUDA build, use `torchvision>=0.21.0=cuda*` instead
- `cuda_compiler>=12.8.1` CUDA compiler for [`torch.compile()`](https://pytorch.org/docs/stable/generated/torch.compile.html) (also installs C and C++ compilers)
- `ipykernel>=6.29.5`: necessary to run Jupyter notebook
- `matplotlib>=3.10.1`: used to create plots
- `pandas>=2.2.3`: used to read csv data

### Flags

- example flags include `-m` to choose the model and `-d` to choose the dataset
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
  - `SZT`: the dataset (included in the repository) used by Schwartz-Ziv & Tishby (2017) in their paper [Opening the Black Box of Deep Neural Networks via Information](https://arxiv.org/abs/1703.00810)
  - `MNIST`: [MNIST](http://yann.lecun.com/exdb/mnist/)
  - `CIFAR10`: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
  - `FashionMNIST`: [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
- for a full list of flags, run `python main.py --help`
- further information is also available in the `get_args()` function of `utils.py`

### Plotting Results

To plot the results, run the cells in the Jupyter notebook `plots.ipynb` using the conda environment from above as the kernel.
Alternatively, any kernel with `ipykernel>=6.29.5`, `matplotlib>=3.10.1`, and `pandas>=2.2.3` installed should also work.

## Code Structure

The code consists of four main Python modules and one Jupyter notebook:

- `utils.py`: contains
  - the command line flag parser
  - a slightly modified version of Algorithm 1 of ["Learning Optimal Representations with the Decodable Information Bottleneck"](https://proceedings.neurips.cc/paper_files/paper/2020/hash/d8ea5f53c1b1eb087ac2e356253395d8-Abstract.html) (see [Modified Algorithm 1](#modified-algorithm-1))
  - the algorithms for computing the NC1 metric and the DIB (see [Algorithms](#algorithms) for more details)
- `main.py`: "glue code" which sets up the dataset(s), then creates and trains the model
- `datasets.py`: logic for loading and transforming the datasets as well as the SZT dataset
- `networks.py`: network architectures
- `plots.ipynb`: notebook for creating plots

When using a dataset for the first time, Lightning will download it into `data/`.
During training run $i$, Lightning stores model checkpoints for the main network in `lightning_logs/version_`$2i$`/checkpoints/`
(as well as duplicate checkpoints in `lightning_logs/version_`$2i+1$`/checkpoints/` due to the nested training) and model checkpoints for the DIB network in `lightning_logs/checkpoints`.
The metrics for each epoch are stored in `lightning_logs/version_`$2i$`/metrics.csv`.

## Algorithms

Both the NC1 and DIB metrics are computed after each epoch.

### Modified Algorithm 1

1. for each class $y$, enumerate the samples $\mathcal X_y = \{x \mid x$ has label $y\}$, i.e., assign them the indices $0,1,…,|\mathcal X_y| - 1$
2. convert each index to base $C$ (the number of classes), implicitly padding with zeros to the left
3. the new labels for each sample are the digits of its base-$C$ representation

<!-- markdownlint-disable MD033 -->
<details>
<summary>Example</summary>

Let $C = 5$ and the samples be labeled [0, 0, 0, 1, 1, 3, 4, 4, 4, 4, 4, 4].
Then, the samples are assigned the indices [0, 1, 2, 0, 1, 0, 0, 1, 2, 3, 4, 5].
Converting to base $C = 5$ and zero-padding to the left gives \[[0, 0], [0, 1], [0, 2], [0, 0], [0, 1], [0, 0], [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 0]\].
The new sample labels are then [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] and [0, 1, 2, 0, 1, 0, 0, 1, 2, 3, 4, 0].

</details>
<!-- markdownlint-enable MD033 -->

### NC1 Computation

- goal: each epoch, compute $\operatorname{tr}(Σ_W^l Σ_B^{l+})$ for all layers $l ∈ ℒ$
- the activations $\{𝐡ˡ_{c,i}\}_{l∈ℒ,\ c ∈ \{1,…,C\},\ i ∈ \{1,…,N\}}$ are accessed by registering forward hooks:
  - MLP: at the end of each nonlinearity
  - ConvNeXt-T, ResNet-18: at the end of each residual block
  - these hooks store the output of each hooked layer after each forward pass
- since the activations don't fit in memory all at once, only store batch activations $\{𝐡ˡ_{c,i}\}_{l ∈ ℒ,\ c ∈ \{1,…,C\},\ i ∈ \{1,…,S\}}$ where $S$ is the batch size
- after training a batch and triggering all forward hooks, update the running
  - class counts $\{n_c\}_{c=1}^C$
  - class totals $\{\{∑_{i=1}^{n_c} 𝐡_{c,i}^l\}_{c=1}^C\}_{l ∈ ℒ}$
  - gram matrices $G^l = ∑_{c=1}^C ∑_{i=1}^{n_c} 𝐡_{c,i}^l 𝐡_{c,i}^{l⊤}$ (useful later)
- computing $Σ_B^l$
  - compute $\boldsymbol μ_c^l = \frac 1{n_c} ∑_{i=1}^{n_c} 𝐡_{c,i}^l$ for $c = 1,…,C$
  - $\bar{\boldsymbol μ}^l = \frac 1N ∑_{c=1}^C ∑_{i=1}^{n_c} 𝐡_{c,i}^l$ where $N = Σ_{c=1}^C n_c$
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
  - finally, compute $\operatorname{tr}(Σ_W^l Σ_B^{l+}) = \operatorname{tr}(Σ_B^{l+} Σ_W^l)$ by solving the least squares problem
    $$ X^* = \min_X \lVert Σ_B^l X - Σ_W^l \rVert_F $$
    and then computing $\operatorname{tr}(X^*)$

### DIB Computation

1. compute new labels for all samples using [modified Algorithm 1](#modified-algorithm-1)
2. split the original network $N$ into an encoder $E$ and decoder $D$
3. for each new labelling of the samples, create a copy of $D$
4. combine the encoder $E$ and decoders into a single model $M$
   <!---->
   ```plaintext
           D
         /
   M = E – ⋮
         \
           D
   ```
   <!---->
5. train $M$ using cross-entropy loss
6. return the final training loss of $M$
7. for each epoch of interest,
   - update the parameters of $E$
   - reset the parameters of $D$
   - repeat steps 2-6
8. repeat steps 2-7 for every (interesting) encoder-decoder split of $N$
