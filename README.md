# Explaining Deep Neural Networks using Information Theory and Geometry

This repository contains the code to reproduce the results of my master's thesis.
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
              ^
              insert --gpus all here to use GPUs
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
  - `MNISTNet`: [Tuomas Oikarinen's optimised CNN](https://github.com/tuomaso/train_mnist_fast)
  - `CIFARNet`: the [CIFAR-10 Airbench architecture](https://github.com/KellerJordan/cifar10-airbench) from [94% on CIFAR-10 in 3.29 Seconds on a Single GPU](https://arxiv.org/abs/2404.00498)
  - `ConvNeXt`: the [ConvNeXt-T architecture](https://pytorch.org/vision/0.21/models/generated/torchvision.models.convnext_tiny.html) from [A ConvNet for the 2020s](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html)
  - `ResNet`: the [ResNet-18 architecture](https://pytorch.org/vision/0.21/models/generated/torchvision.models.resnet18.html) from [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
- supported datasets (case-sensitive):
  - `CIFAR10`: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
  - `FashionMNIST`: [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
  - `MNIST`: [MNIST](http://yann.lecun.com/exdb/mnist/)
  - `SZT`: the dataset (included in the repository) used by [Schwartz-Ziv & Tishby (2017)](https://arxiv.org/abs/1703.00810)
- for a full list of flags, run `python main.py --help`
- further information is also available in the `get_args()` function of `utils.py`

### Plotting Results

To plot the results, run the cells in the Jupyter notebook `plots.ipynb` using the conda environment from above as the kernel.
Alternatively, any kernel with `ipykernel>=6.29.5`, `matplotlib>=3.10.1`, and `pandas>=2.2.3` installed should also work.

## Code Structure

The code consists of four main Python modules and one Jupyter notebook:

- `datasets.py`: logic for loading and transforming the datasets, also contains a slightly modified version of Algorithm 1 of [Learning Optimal Representations with the Decodable Information Bottleneck](https://proceedings.neurips.cc/paper_files/paper/2020/hash/d8ea5f53c1b1eb087ac2e356253395d8-Abstract.html) (see [Modified Algorithm 1](#modified-algorithm-1))
- `main.py`: "glue code" which
  - sets up the dataset(s),
  - uses an (optional) learning rate (LR) tuner based on the LR range test of [Cyclical Learning Rates for Training Neural Networks](https://ieeexplore.ieee.org/document/7926641), and
  - creates and trains the model
- `networks.py`: network architectures
- `utils.py`: contains the command line flag parser and the algorithms for computing the NC1 metric and the DIB (see [Algorithms](#algorithms) for more details)
- `plots.ipynb`: notebook for creating plots

When using a dataset for the first time, Lightning will download it into the directory specified by `--data-dir` (default: `data/`).
Lightning stores model checkpoints for the main network in `lightning_logs/version_X/checkpoints/` and model checkpoints for the DIB network in `lightning_logs/checkpoints`.
Hyperparameters are stored in `lightning_logs/version_X/hparams.yaml` and the metrics are stored in `lightning_logs/version_X/metrics.csv`.

## Algorithms

Both the NC1 and DIB metrics are computed after each epoch.

### Modified Algorithm 1

1. For each class $y$, enumerate the samples $\mathcal X_y = \{x ∣ x$ has label $y\}$, i.e., assign them the indices $0,1,…,|\mathcal X_y| - 1$
2. Convert each index to base $C$ (the number of classes), implicitly padding with zeros to the left. It is easy to see that the maximum number of digits is $⌈\log_{|Y|}(\max\{|\mathcal X_y| : y ∈ Y\})⌉$.
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
- The activations $\{𝐡ˡ_{c,i}\}_{l∈ℒ,\ c ∈ \{1,…,C\},\ i ∈ \{1,…,N\}}$ are accessed by registering forward hooks at the penultimate layer as well as:
  - MLP: after each nonlinearity
  - MNISTNet: after each convolutional block
  - CIFARNet: after each downsampling layer
  - ConvNeXt-T, ResNet-18: after each residual block
  - these hooks store the output of each hooked layer after each forward pass
- Since the activations don't fit in memory all at once, only the batch activations $\{𝐡ˡ_{c,i}\}_{l ∈ ℒ,\ c ∈ \{1,…,C\},\ i ∈ \{b_1,…,b_S\}}$ can be used where $b_n$ is element $n$ of batch $b$ and $S$ is the batch size.
- Furthermore, for (large) convolutional neural networks (CNNs), $𝐡ˡ_{c,i} \in ℝ^{d≤10⁵}$, which makes computing $Σ_W^l ∈ ℝ^{d×d}$ and $Σ_B^l ∈ ℝ^{d×d}$ directly undesirable.

<!-- omit in toc -->
#### The Algorithm

The algorithm requires *two* passes over the dataset.

0. Before training, compute the class counts $\{n_c\}_{c=1}^C$.
   This is possible since the labels $(∈ℕ^{≈10⁵})$ fit in memory.
1. (**first pass**): After each batch, update the running class totals $\{\{∑_{i=1}^{n_c} 𝐡_{c,i}^l\}_{c=1}^C\}_{l ∈ ℒ}$.
2. Using the final class totals,
   - compute $\boldsymbol μ_c^l = \frac 1{n_c} ∑_{i=1}^{n_c} 𝐡_{c,i}^l$ for $c = 1,…,C$
   - compute $\bar{\boldsymbol μ}ˡ = \frac 1N ∑_{c=1}^C ∑_{i=1}^{n_c} 𝐡_{c,i}^l$ where $N = Σ_{c=1}^C n_c$
   - compute $𝐌ˡ = [\boldsymbol μ₁ˡ - \bar{\boldsymbol μ}ˡ, ⋯, \boldsymbol μ_C^l - \bar{\boldsymbol μ}ˡ]$, recall: $Σ_B^l = \frac1C 𝐌ˡ(𝐌ˡ)^⊤$
   - since $𝐌ˡ(𝐌ˡ)^⊤$ and $(𝐌ˡ)^⊤𝐌ˡ$ share eigenvalues (see proof of step 4), compute the eigendecomposition of (the much smaller) $(𝐌ˡ)^⊤𝐌ˡ = 𝐕𝚲𝐕^⊤$
3. (**second pass**): After each batch, update the running sum $∑_{c=1}^C ∑_{i=1}^{n_c} \left(\frac{(κ_{c,i})_j}{λ_j}\right)²$ where $κ_{c,i} = 𝐕^⊤(𝐌ˡ)^⊤(𝐡_{c,i}^l - \boldsymbol μ_c^l) ∈ ℝ^C$ and $𝚲 = \operatorname{diag}(λ₁, …, λ_C)$.
4. Finally, $\operatorname{tr}(Σ_W^l Σ_B^{l+}) = \frac CN ∑_{c=1}^C ∑_{i=1}^{n_c} \left(\frac{(κ_{c,i})_j}{λ_j}\right)²$

<!-- markdownlint-disable MD033 -->
<details>
<summary>Proof of Step 4</summary>

Consider the SVD of $𝐌ˡ = 𝐔𝐒𝐕^⊤$. We then have $𝐌ˡ𝐕𝐒^{-1} = 𝐔$ and
$$
  𝐌ˡ(𝐌ˡ)^⊤ = 𝐔𝐒𝐕^⊤𝐕𝐒𝐔^⊤ = 𝐔𝐒²𝐔^⊤ = 𝐔𝚲𝐔^⊤ \\
(𝐌ˡ)^⊤𝐌ˡ = 𝐕𝐒𝐔^⊤𝐔𝐒𝐕^⊤ = 𝐕𝐒²𝐕^⊤ = 𝐕𝚲𝐕^⊤
$$
By definition of the pseudoinverse, since $Σ_B^l = \frac1C 𝐌ˡ(𝐌ˡ)^⊤ = 𝐔(𝚲/C)𝐔^⊤$,
$Σ_B^{l+} = 𝐔(𝚲/C)⁺𝐔^⊤ = C𝐔𝚲^{-1}𝐔^⊤$.
Now, since $\operatorname{tr}(𝐚𝐛^⊤) = 𝐛^⊤𝐚$ for vectors $𝐚,𝐛$, we have
$$
  \operatorname{tr}(Σ_W^l Σ_B^{l+}) = \frac1N \operatorname{tr}((𝐡_{c,i}^l - \boldsymbol μ_c^l)(𝐡_{c,i}^l - 𝛍_c^l)^⊤) \\
$$

</details>
<!-- markdownlint-enable MD033 -->

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
7. for each epoch,
   - update the parameters of $E$ with those of $N$
   - reset the parameters of $D$
   - repeat steps 2-6
8. repeat steps 2-7 for every (interesting) encoder-decoder split of $N$
