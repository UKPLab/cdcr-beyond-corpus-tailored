FROM ubuntu:bionic

RUN apt update && \
    apt install -y openjdk-8-jre-headless make python3.7 python3.7-venv python3.7-dev build-essential nano vim less screen tmux unzip wget locales && \
    locale-gen en_US.UTF-8
COPY . /cdcr
WORKDIR /cdcr

# set locale to make python Click happy
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8

ENTRYPOINT ["/bin/bash"]