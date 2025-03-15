FROM nvcr.io/nvidia/pytorch:22.12-py3

RUN apt-get update && apt-get upgrade -y && \
    apt-get install vim git tmux sudo -y 

RUN pip install --upgrade pip

RUN pip install gpustat

WORKDIR /

ARG UNAME
ARG UID
ARG GID
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME

RUN echo "$UNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER $UNAME