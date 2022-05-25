# HyperCoref Data Post-Processing
Code in this branch performs data postprocessing for converting the [HyperCoref corpus](https://github.com/UKPLab/emnlp2021-hypercoref-cdcr/) (and the FCC-T and GVC corpora) into a format suitable for training/testing the CDCR system of Cattan et al. 2020.
It also collects statistics on many aspects of CDCR corpora such as cluster size distribution, mention length, lexical variation, and more.

## Contents
- [Recommended System Specs](#recommended-system-specs)
- [Docker Preparation](#docker-preparation)
- [Data Gathering](#data-gathering)
- [Usage](#usage)
    - [S\_gg train/dev/test for FCC-T, GVC](#s_gg-traindevtest-for-fcc-t-gvc)
    - [S\_ss train/dev and S\_gg\_aug train for ABC, BBC](#s_ss-traindev-and-s_gg_aug-train-for-abc-bbc)
        - [Trial (wired.com)](#trial-wiredcom)
        - [ABC/BBC](#abcbbc)
    - [Where to Find the Resulting Postprocessed Corpora](#where-to-find-the-resulting-postprocessed-corpora)
    - [Statistics for All Corpora and the Entire HyperCoref](#statistics-for-all-corpora-and-the-entire-hypercoref)
- [Troubleshooting](#troubleshooting)

## Recommended System Specs
We recommend using a machine with:
* at least 4 logical CPU cores
* at least 16GB of RAM
* 500GB of free disk space
* installed [Docker](https://docs.docker.com/engine/install/) and [docker-compose](https://docs.docker.com/compose/install/)

## Docker Preparation
1. Clone this git repository on the `hypercoref` branch: `git clone --depth 1 --branch hypercoref https://github.com/UKPLab/cdcr-beyond-corpus-tailored/`
2. Inside the `.env` file, specify the location of the HyperCoref corpus on the host system (default is `~/hypercoref`) and the location to which the postprocessed data will be written (default `~/cdcr`).
3. On the host, run `mkdir -p ~/cdcr/data` (adjust path to what you specified in `.env`.).
4. Run `docker-compose up -d` in the folder containing this readme. Apart from spawning a CoreNLP service, this spawns a container called `cdcr-data-postpro-hypercoref-pipeline` in which our scripts are meant to be run:
    * Attach to the interactive shell in this container by running `docker attach cdcr-data-postpro-hypercoref-pipeline`.
    * To detach it (without stopping the container), press `Ctrl+p, Ctrl+q` in sequence.

## Data Gathering
This project supports the following corpora:

| Corpus | How to obtain |
|--------|---------------|
| ECB+   | See [master branch of this repo](https://github.com/UKPLab/cdcr-beyond-corpus-tailored/) |
| GVC | See [master branch of this repo](https://github.com/UKPLab/cdcr-beyond-corpus-tailored/) |
| FCC / FCC-T | See https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2305 |
| HyperCoref | See https://github.com/UKPLab/emnlp2021-hypercoref-cdcr/tree/master/hypercoref |

After obtaining all or some of these corpora, the directory tree of `~/cdcr/data` should look like this:
```
~/cdcr/data
├── ecbplus
│   ├── ECB+_LREC2014
│   ├── ECB+_LREC2014.zip
│   ├── ECBplus_coreference_sentences.csv
│   ├── guns
│   ├── sports
│   ├── test
│   ├── train
│   └── valid
├── football
│   ├── 2020-03-18_FCC
│   ├── 2020-09-11_ECBplus_publication_dates
│   ├── 2020-10-05_FCC_cleaned
│   └── 2020-10-05_FCC-T
├── gun_violence
│   ├── dev.csv
│   ├── gvc_doc_to_event.csv
│   ├── GVC_gold.conll
│   ├── test.csv
│   └── train.csv
└── hypercoref
    └── ...
```

## Usage
Our paper contains experiments in three scenarios with different amounts of gold-standard data (`S_gg`, `S_sg`, `S_ss`) as well as gold-standard training data augmented with HyperCoref data (`S_gg_aug`).

All commands are to be run in the `cdcr-data-postpro-hypercoref-pipeline` docker container.

### S_gg train/dev/test for FCC-T, GVC
To create train/dev/test splits of the FCC-T and GVC corpora in the Cattan et al. format, run:
```bash
find resources/yaml/data_preprocessing/cattan_fmt/gg -name *.yaml -exec python3 run_pipeline.py run resources/yaml/device_docker.yaml {} \;
```

For ECB+, we used the files provided by Cattan et al.

### S_ss train/dev and S_gg_aug train for ABC, BBC
We explain the steps using wired.com from HyperCoref instead of ABC/BBC because the former is smaller, and therefore faster for confirming that the data pipeline is working as intended. The steps for postprocessing ABC/BBC are analogous and outlined below.

#### Trial (wired.com)
1. Having obtained HyperCoref parquet files (following [the hypercoref github repo](https://github.com/UKPLab/emnlp2021-hypercoref-cdcr)), copy the resulting parquet files into the right location for this project (adjust `HASH` and `DATE` in the command):
   ```bash
   cp -r /hypercoref/article_extraction_wired.com/HASH/DATE/7_CreateSplitsStage_create_splits/* /cdcr/resources/data/hypercoref/wired.com/
   ```
2. Compute corpus statistics for wired.com:
   ```bash
   python3 run_pipeline.py run resources/yaml/device_docker.yaml resources/yaml/data_preprocessing/stats/hypercoref/wired.com.yaml
   ```
   This writes constituency and dependency parse trees to disk as a byproduct, which are needed to convert long hyperlink anchor texts in HyperCoref into minimum-span annotations for CDCR systems in step 4.
3. Copy the files containing the parse trees into the correct data folder (adjust `HASH` and `DATE` in the command):
   ```bash
   cp /cdcr/working_dir/hypercoref_stats_wired.com/HASH/DATE/2_CoreNlpProcessorStage/* /cdcr/resources/data/hypercoref/wired.com/
   ```
4. Run the data pipeline for all splits involving wired.com:
   ```bash
   find resources/yaml/data_preprocessing/cattan_fmt/ -name *wired*.yaml -exec python3 run_pipeline.py run resources/yaml/device_docker.yaml {} \;
   ```

#### ABC/BBC
1. Copy parquet files into the right location (adjust `HASH` and `DATE` in each command):
   ```bash
   cp -r /hypercoref/article_extraction_abcnews.go.com/HASH/DATE/7_CreateSplitsStage_create_splits/* /cdcr/resources/data/hypercoref/abcnews.go.com/
   ```
   and
   ```bash
   cp -r /hypercoref/article_extraction_bbc.com/HASH/DATE/7_CreateSplitsStage_create_splits/* /cdcr/resources/data/hypercoref/bbc.com/
   ```
1. Compute corpus statistics:
   ```bash
   python3 run_pipeline.py run resources/yaml/device_docker.yaml resources/yaml/data_preprocessing/stats/hypercoref/abcnews.go.com.yaml
   python3 run_pipeline.py run resources/yaml/device_docker.yaml resources/yaml/data_preprocessing/stats/hypercoref/bbc.com.yaml
   ```
3. Copy the files containing constituency & dependency parse trees into the correct data folder (see the trial instructions above).
4. Run postprocessing:
   ```bash
   find resources/yaml/data_preprocessing/cattan_fmt/ -name *.yaml -exec python3 run_pipeline.py run resources/yaml/device_docker.yaml {} \;
   ```

### Where to Find the Resulting Postprocessed Corpora
On the host machine, `~/cdcr/working_dir/preprocess_XYZ/HASH/DATE/W_DatasetExporterStage/cattan/DATASET_NAME/SPLIT/` will contain the relevant files to use with the Cattan et al. system: `{SPLIT}.json`, `{SPLIT}_corpus_level.conll`, and `{SPLIT}_events.json`.

`XYZ`, `HASH`, `DATE`, `W`, `DATASET_NAME`, `SPLIT` all depend on the inputs the pipeline was run with.

### Statistics for All Corpora and the Entire HyperCoref
1. Run
   ```bash
   find resources/yaml/data_preprocessing/stats/ -name *.yaml -exec python3 run_pipeline.py run resources/yaml/device_docker.yaml {} \;
   ```

## Troubleshooting
* If the pipeline fails with `illegal instruction (core dumped)` in `HyperlinksHackStage`, then the pre-built [kahypar](https://github.com/kahypar/kahypar/) binaries are not compatible with your system.
    * To fix this, [build the kahypar python module from source](https://github.com/kahypar/kahypar#the-python-interface) and try replacing the binary [in your site-packages](https://stackoverflow.com/a/46071447) installed via pip with the self-built one.
* Please open a github issue in case of other problems.
