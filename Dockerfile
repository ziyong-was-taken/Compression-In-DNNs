FROM mambaorg/micromamba:latest

COPY environment.yaml /tmp/env.yaml
# COPY *.py .
COPY data/SZT.pt data/SZT.pt

ENV CONDA_OVERRIDE_CUDA=12.8

RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes
