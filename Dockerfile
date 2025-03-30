FROM mambaorg/micromamba:latest

COPY environment.yaml /tmp/env.yaml
COPY *.py .
COPY data/SZT.pt data/SZT.pt

# use CUDA 12.6 for solver
ENV CONDA_OVERRIDE_CUDA=12.6

RUN micromamba install -y -n base -f /tmp/env.yaml
RUN micromamba clean -afy