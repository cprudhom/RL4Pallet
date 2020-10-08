# Distributed under the terms of the Modified BSD License.
ARG BASE_CONTAINER=jupyter/scipy-notebook
FROM $BASE_CONTAINER

LABEL maintainer="Charles Prud'homme <charles.prudhomme@imt-atlantique.fr>"

USER root

# Install OpenAI Gym
RUN pip install gym

# Install facets which does not have a pip or conda package at the moment
WORKDIR $HOME
RUN git clone https://github.com/cprudhom/RL4Pallet.git && \
    cd RL4Pallet && \
    rm ./learn_advanced.py ./learn_qtable.py ./learn_dummy.py ./Dockerfile && \
    cd .. && \
    mv RL4Pallet/* ./  && \
    pip install -e .  && \
    rm -r RL4Pallet && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

USER $NB_UID

WORKDIR $HOME
