#!/usr/bin/env bash

readonly ALLENNLP_OLD_VENV=venv_allennlp_0.9.0
readonly ALLENNLP_NEW_VENV=venv_allennlp_1.0.0

CORPUS="fcct_gvc"

for split in "train" "dev" "traindev" ; do
    source ${ALLENNLP_OLD_VENV}/bin/activate
    python3 run_pipeline.py run resources/yaml/device_docker.yaml resources/yaml/data_preprocessing/cross_dataset/${CORPUS}_${split}_pt1_allennlp0.9.0.yaml
    deactivate
    source ${ALLENNLP_NEW_VENV}/bin/activate
    python3 run_pipeline.py run resources/yaml/device_docker.yaml resources/yaml/data_preprocessing/cross_dataset/${CORPUS}_${split}_pt2_allennlp1.0.0.yaml
    deactivate
done