# syntax=docker/dockerfile:1

FROM python:3.7.12-bullseye

RUN apt update && \
    apt install -y make build-essential nano vim less screen tmux unzip wget locales && \
    locale-gen en_US.UTF-8
COPY . /cdcr
WORKDIR /cdcr

# set locale to make python Click happy
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8

# python dependencies
RUN pip install --upgrade pip==21.3.1 wheel && \
    pip install -r /cdcr/resources/requirements/base.txt

ENTRYPOINT ["/bin/bash"]