#!/usr/bin/env bash

readonly ALLENNLP_OLD_VENV=venv_allennlp_0.9.0
readonly ALLENNLP_NEW_VENV=venv_allennlp_1.0.0

if [ "$1" == "" ]; then
    echo "A corpus must be specified (ecbp/fcct/gvc)"; exit 1
fi
CORPUS=$1

for split in "train" "dev" "test" "traindev" ; do
    if [ $CORPUS != "fcct" ]; then
        source ${ALLENNLP_OLD_VENV}/bin/activate
        python3 run_pipeline.py run resources/yaml/device_docker.yaml resources/yaml/data_preprocessing/${CORPUS}/${CORPUS}_${split}_pt1_allennlp0.9.0.yaml
        deactivate
    fi
    source ${ALLENNLP_NEW_VENV}/bin/activate
    python3 run_pipeline.py run resources/yaml/device_docker.yaml resources/yaml/data_preprocessing/${CORPUS}/${CORPUS}_${split}_pt2_allennlp1.0.0.yaml
    deactivate
done