# syntax=docker/dockerfile:1

FROM debian:bullseye

RUN apt update && apt install -y openjdk-17-jre-headless wget unzip
COPY resources/scripts /corenlp
WORKDIR /corenlp

# CoreNLP: unless the ZIP file is provided in the build context, download it fresh (which takes quite some time)
ARG CORENLP_URL="https://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip"
RUN if [ ! -f "stanford-corenlp-full-2018-10-05.zip" ]; then \
      wget -c ${CORENLP_URL}; \
    fi && \
    unzip -q *.zip -d . && \
    rm *.zip

ENTRYPOINT ["/bin/bash", "/corenlp/docker_run_corenlp.sh"]
EXPOSE 9000