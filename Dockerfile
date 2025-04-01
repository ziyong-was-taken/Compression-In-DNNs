FROM mambaorg/micromamba:latest

COPY environment.yaml /tmp/env.yaml
COPY *.py .
COPY data/SZT.pt data/SZT.pt

# use CUDA 12.6 for solver
ENV CONDA_OVERRIDE_CUDA=12.6

# install c++ compiler
USER root
RUN apt-get update
RUN apt-get install -y g++
USER mambauser

# install environment
RUN micromamba install -y -n base -f /tmp/env.yaml
RUN micromamba clean -afy