# Distributed under the terms of the Modified BSD License.
ARG BASE_CONTAINER=jupyter/scipy-notebook
FROM $BASE_CONTAINER

LABEL maintainer="Charles Prud'homme <charles.prudhomme@imt-atlantique.fr>"

USER root

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update \
 && apt-get install -yq --no-install-recommends \
    python-opengl \
 && apt-get clean

# Install OpenAI Gym
RUN pip install gym

# Install facets which does not have a pip or conda package at the moment
WORKDIR $HOME
RUN git clone https://github.com/cprudhom/RL4Pallet.git && \
    cd RL4Pallet && \
    rm -r ./Dockerfile ./NOM_Prenom.ipynb && \
    cd .. && \
    mv RL4Pallet/* ./  && \
    pip install -e .  && \
    rm -r RL4Pallet && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}" && \
    python3 ./learn_dummy.py

USER $NB_UID

WORKDIR $HOME
