# Generalizing Cross-Document Event Coreference Resolution Across Multiple Corpora
This repository contains the code for reproducing the results of our [Computational Linguistics article](https://direct.mit.edu/coli/article/47/3/575/102774/Generalizing-Cross-Document-Event-Coreference) which was presented at [EMNLP 2021](https://2021.emnlp.org/).

It contains:
- our cross-document event coreference resolution (CDCR) system implementation
- the corresponding corpus preprocessing pipeline for the ECB+, Gun Violence Corpus (GVC) and Football Coreference Corpus (FCC) corpora
- baseline implementations
- [CoNLL files and scores from our experiments](archive/)
- [the annotation guidelines used](guidelines/)

For obtaining the **Football Coreference Corpus (FCC)**, please visit https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2305

Please cite our work as follows:
```
@article{bugert2021crossdocument,
    author = {Bugert, Michael and Reimers, Nils and Gurevych, Iryna},
    title = {{Generalizing Cross-Document Event Coreference Resolution Across Multiple Corpora}},
    journal = {Computational Linguistics},
    volume = {47},
    number = {3},
    pages = {575-614},
    year = {2021},
    month = {11},
    issn = {0891-2017},
    doi = {10.1162/coli_a_00407},
    url = {https://doi.org/10.1162/coli_a_00407},
    eprint = {https://direct.mit.edu/coli/article-pdf/47/3/575/1971857/coli_a_00407.pdf},
}
```

> **Abstract:** Cross-document event coreference resolution (CDCR) is an NLP task in which mentions of events need to be identified and clustered throughout a collection of documents. CDCR aims to benefit downstream multidocument applications, but despite recent progress on corpora and system development, downstream improvements from applying CDCR have not been shown yet. We make the observation that every CDCR system to date was developed, trained, and tested only on a single respective corpus. This raises strong concerns on their generalizability—a must-have for downstream applications where the magnitude of domains or event mentions is likely to exceed those found in a curated corpus. To investigate this assumption, we define a uniform evaluation setup involving three CDCR corpora: ECB+, the Gun Violence Corpus, and the Football Coreference Corpus (which we reannotate on token level to make our analysis possible). We compare a corpus-independent, feature-based system against a recent neural system developed for ECB+. Although being inferior in absolute numbers, the feature-based system shows more consistent performance across all corpora whereas the neural system is hit-or-miss. Via model introspection, we find that the importance of event actions, event time, and so forth, for resolving coreference in practice varies greatly between the corpora. Additional analysis shows that several systems overfit on the structure of the ECB+ corpus. We conclude with recommendations on how to achieve generally applicable CDCR systems in the future—the most important being that evaluation on multiple CDCR corpora is strongly necessary. To facilitate future research, we release our dataset, annotation guidelines, and system implementation to the public.

Contact person: Michael Bugert

https://ukp.tu-darmstadt.de

https://tu-darmstadt.de

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.

## Contents

- [Contents](#contents)
- [Setup](#setup)
    - [Requirements](#requirements)
    - [Preparatory steps](#preparatory-steps)
- [Reproducing our results](#reproducing-our-results)
    - [ECB+](#ecb)
    - [FCC-T and GVC](#fcc-t-and-gvc)
- [Instructions for other experiments](#instructions-for-other-experiments)
    - [Corpus preprocessing options](#corpus-preprocessing-options)
        - [Merging corpora](#merging-corpora)
        - [Reducing corpus size for debugging](#reducing-corpus-size-for-debugging)
        - [Masking mentions](#masking-mentions)
        - [Exporting corpus statistics](#exporting-corpus-statistics)
    - [Model implementation](#model-implementation)
        - [Feature selection](#feature-selection)
        - [Hyperparameter optimization](#hyperparameter-optimization)
        - [Document preclustering at test time](#document-preclustering-at-test-time)
        - [Separate training/evaluation of the mention pair classifier](#separate-trainingevaluation-of-the-mention-pair-classifier)
        - [Running baseline experiments](#running-baseline-experiments)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## Setup

### Requirements
You need:
* a machine with [Docker](https://www.docker.com/) installed
* at least 140GB of free disk space at the location where the Docker container will end up (`/var/lib/docker` by default)
* For running hyperparameter optimization:
    * A powerful CPU compute server (the more cores and RAM, the better).
    * Fast temporary storage which can accomodate several hundred GB of cached features. Its location doesn't have to be `/tmp` (see [python tempfile docs](https://docs.python.org/3.7/library/tempfile.html#tempfile.gettempdir)). For your options with temporary storage and Docker, see for example [this post on stackoverflow](https://stackoverflow.com/a/55104489). 

Not required, but recommended:
* A self-hosted instance of the **DBpedia Spotlight** entity linker, see https://github.com/dbpedia-spotlight/dbpedia-spotlight/wiki/Installation. We used the [English model from 2020-03-11](http://downloads.dbpedia.org/repo/dbpedia/spotlight/spotlight-model/2020.03.11/). If you don't want to host it yourself you can use the public endpoint, but be aware that it is rate limited.
* A self-hosted instance of **DBpedia**. We used the [latest release as of April 2020](https://databus.dbpedia.org/dbpedia/collections/latest-core) and imported it into a [Virtuoso instance](http://vos.openlinksw.com/owiki/wiki/VOS). The alternative to self-hosting is using [the public endpoint](https://wiki.dbpedia.org/public-sparql-endpoint) which is rate limited and (at the time of writing) runs an older release from October 2016. 

Please note that using the public endpoints will produce preprocessed corpora different from the ones we used in our experiments, hence scores obtained with the public endpoints will not match those reported in the paper.
[We provide our preprocessed corpora for reference.](archive/)

### Preparatory steps
1. Open a terminal and run
   ```bash
   docker run --name cdcr-container -it mbugert/cdcr-beyond-corpus-tailored
   ```
   This opens a shell in a docker container in which all experiments can be performed. To detach it (without stopping the container), press Ctrl+p, Ctrl+q in sequence. Use `docker attach cdcr-container` to re-attach.
2. In the container, run
    ```bash
   make -f resources/scripts/prepare.makefile
    ```
    What this does:
    * It downloads [Pytorch-Biggraph](https://github.com/facebookresearch/PyTorch-BigGraph) embeddings for Wikidata.
    * It downloads and prepares SpanBERT embeddings pretrained for AllenNLP for the use with our model.
    * It prepares two virtualenvironments, one with allennlp 0.9.0 and one with allennlp 1.0.0. This is unfortunately necessary because the pretrained SRL system and the pretrained SpanBERT system used during preprocessing require different allennlp versions.
    * It downloads and prepares the ECB+ and GVC corpora.

3. To prepare the FCC-T corpus, visit https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2305 and follow the instructions there.
   Assuming the output of the corpus generation script is `/home/user/datasets`, copy the output into the docker container by running `docker cp /home/user/datasets/. cdcr-container:/cdcr/resources/data/football`.

## Reproducing our results
We explain how to reproduce the results of our feature-based CDCR model on ECB+, FCC-T and GVC reported in table 5 in the paper. All steps should be performed in the docker container.


### ECB+
1. Edit `/cdcr/resources/yaml/device_docker.yaml` (`nano` and `vim` are available).
    * Set `max_cores` to the number of logical CPU cores on your machine.
    * If you have self-hosted instances of DBpedia Spotlight and DBpedia available, you can enter these in the `dbpedia` and `dbpedia_spotlight` sections. 
2. Before being used in model experiments, each corpus requires preprocessing. To do this for ECB+, run:
   ```
   cd /cdcr
   resources/scripts/run_data_preprocessing.sh ecbp
   ```
   This will perform preprocessing twice for each corpus split: once using allennlp 0.9.0 for predicting semantic roles (the results are cached to disk), then a second time with allennlp 1.0.0 for preparing embedding features and for exporting the resulting split into a pickle file. 
   Upon completion, `/cdcr/working_dir` will contain a folder structure similar to the one shown below (hexadecimal config hashes and timestamps will be different on your end). 
   ```
   /cdcr/working_dir/
   ├── global
   ├── preprocess_ECBP_dev
   │   ├── 478ea5b0
   │   └── b6072997
   │       └── 2020-11-03_16-52-51
   |           ├── ...
   │           └── 9_DatasetExporterStage
   │               └── ecbp_dev_preprocessed_b6072997_2020-11-03_16-52-51.pickle
   ├── preprocess_ECBP_test
   │   ├── bf1a5dab
   │   │   └── 2020-11-03_17-06-35
   |   |       ├── ...
   │   │       └── 9_DatasetExporterStage
   │   │           └── ecbp_test_preprocessed_bf1a5dab_2020-11-03_17-06-35.pickle
   │   └── ef8f28d8
   └── preprocess_ECBP_train
   |   ├── 115b6261
   |   │   └── 2020-11-03_16-14-03
   |   │       ├── ...
   |   │       └── 9_DatasetExporterStage
   |   │           └── ecbp_train_preprocessed_115b6261_2020-11-03_16-14-03.pickle
   |   └── b4739c2c
   └── preprocess_ECBP_traindev
       ├── 20b39565
       |   └── 2020-11-05_16-02-19
       |       ├── ...
       |       └── 10_DatasetExporterStage
       |           └── ecbp_traindev_preprocessed_20b39565_2020-11-05_16-02-19.pickle
       └── 2b97be94
   ```
3. To train on ECB+:
    1. First, edit `/cdcr/resources/yaml/train/ecbp/ecbp_clustering_xgboost.yaml`.
        * This file is pre-filled for training five models (with different random seeds) using XGBoost, with optimal features and hyperparameters for ECB+.
        * Fill in the `train_data_path` variable with the path to the `ecbp_traindev_preprocessed_*.pickle` file created in step 2.
    2. Start the training process by running
       ```bash
       cd /cdcr
       source venv_allennlp_1.0.0/bin/activate
       python run_feature_model.py train resources/yaml/device_docker.yaml resources/yaml/train/ecbp/ecbp_clustering_xgboost.yaml
       ```
       Once complete, the serialized models will be located at `/cdcr/working_dir/ecbp_training_clustering_xgboost/[config hash]/[timestamp]/serialized_models`. Take note of this path for the next step.
4. To predict and evaluate the trained model(s) on the ECB+ test split:
    1. Edit `/cdcr/resources/yaml/evaluate/ecbp/ecbp_clustering.yaml` and fill in `eval_data_path` with the path to the `ecbp_test_preprocessed_*.pickle` file created in step 2. 
    2. Run (with `/path/to/serialized_models` set to the path from step 3):
       ```bash
       python run_feature_model.py evaluate /path/to/serialized_models resources/yaml/device_docker.yaml resources/yaml/evaluate/ecbp/ecbp_clustering.yaml
       ```
       This triggers prediction and evaluation with [the coreference evaluation tool of Moosavi et al. 2019](https://github.com/ns-moosavi/coval). The following files are generated:
       ```
       /cdcr/working_dir/ecbp_evaluate_clustering/
       └── 4907b0d0
           └── 2020-11-05_14-50-01
               ├── 0                               # evaluation of the model trained with random seed 0
               │   ├── cross_doc                   # evaluation scenario where all documents are merged into a single meta-document in the gold and system CoNLL files. This is what we report in the paper.
               │   │   ├── eval_cross_doc.txt      # output from the scoring script
               │   │   ├── gold.conll              # gold annotations - the naming scheme for mentions is `metadoc_{document id}_{mention id}`
               │   │   └── system.conll            # predictions
               │   └── within_doc                  # evaluation scenario where documents are scored in isolation, i.e. only within-document links are scored and cross-document links are left out entirely.
               │       ├── eval_within_doc.txt     # output from the scoring script
               │       ├── gold.conll              # gold annotations - the naming scheme for mentions is `{document id}_{mention id}`
               │       └── system.conll            # predictions
               ├── 1                               # evaluation of the model trained with random seed 1, and so on
               │   └── ...
               ├── ...
               ├── metrics_unaggregated.csv        # all scores from all random seeds collected in one file
               ├── metrics_aggregated.csv          # all scores aggregated over the random seeds: the mean/std/min/max achieved for P, R, F1 of MUC, B3, CEAFe, CoNLL, LEA for the cross_doc (`meta_doc=True`) and within_doc (`meta_doc=False`) scenarios
               └── metrics_aggregated_pretty.txt   # the above in human-readable form: the format of the first column is `(meta-doc, metric)`
       ```
       The content of `metrics_aggregated_pretty.txt` is logged to stdout upon completion of the script.

### FCC-T and GVC
The steps for reproducing our FCC-T and GVC results differ only slightly from those reported above [for ECB+](#ecb).
1. Run `export CORPUS=fcct` or `export CORPUS=gvc`.
2. Run preprocessing via `cd /cdcr; resources/scripts/run_data_preprocessing.sh ${CORPUS}`
3. To train:
    1. Edit `/cdcr/resources/yaml/train/${CORPUS}/${CORPUS}_clustering_xgboost.yaml` and fill in `train_data_path` with the path to `${CORPUS}_traindev_preprocessed_*.pickle` created in step 2.
    2. Run
       ```bash
       cd /cdcr
       source venv_allennlp_1.0.0/bin/activate
       python run_feature_model.py train resources/yaml/device_docker.yaml resources/yaml/train/${CORPUS}/${CORPUS}_clustering_xgboost.yaml
       ```
       Once complete, the serialized models will be located at `/cdcr/working_dir/${CORPUS}_training_clustering_xgboost/[config hash]/[timestamp]/serialized_models`.
4. To evaluate:
    1. Edit `/cdcr/resources/yaml/evaluate/${CORPUS}/${CORPUS}_clustering.yaml` and fill in `eval_data_path` with the path to the `${CORPUS}_test_preprocessed_*.pickle` file created in step 2.
    2. Run (with `/path/to/serialized_models` set to the path produced by step 3):
       ```bash
       python run_feature_model.py evaluate /path/to/serialized_models resources/yaml/device_docker.yaml resources/yaml/evaluate/${CORPUS}/${CORPUS}_clustering.yaml
       ```
    3. The results appear inside `/cdcr/working_dir/${CORPUS}_evaluate_clustering/`.

## Instructions for other experiments

### Corpus preprocessing options

#### Merging corpora
To perform joint training over multiple corpora (see section 8 in the paper), one needs to create pickle files containing multiple corpora.
This can be achieved by chaining multiple `python.handwritten_baseline.pipeline.data.loader` sections in the `pipeline` section of a preprocessing YAML config.

Example YAML files for merging the FCC-T and GVC corpora are provided in [`/cdcr/resources/yaml/data_preprocessing/cross_dataset`](resources/yaml/data_preprocessing/cross_dataset): there are six files, two for each split (`train`, `dev` and `train+dev`) to account for allennlp version differences.
Run `cd /cdcr; resources/scripts/run_fcct_gvc_cross-dataset_preprocessing.sh` to create the FCC-T+GVC pickle files in `/cdcr/working_dir/preprocess_FCC-T_GVC_[split]/`. These files can be used interchangeably in place of other pickle files for training/predicting/optimizing the model.

#### Reducing corpus size for debugging
It can be useful to reduce the size of a corpus for speeding up debugging sessions. There is a `python.handwritten_baseline.pipeline.data.processing.reducer` preprocessing pipeline stage which can be used for this purpose. See for example [`/cdcr/resources/yaml/data_preprocessing/ecbp/ecbp_dev_pt1_allennlp0.9.0.yaml`](resources/yaml/data_preprocessing/ecbp/ecbp_dev_pt1_allennlp0.9.0.yaml) which contains a commented out configuration for this pipeline stage.  

#### Masking mentions
The masking of action/participant/time/location mentions or the document publication date (see experiments in section 7.2 in the paper) is achieved with an extra pipeline stage during corpus preprocessing. See for example [`/cdcr/resources/yaml/data_preprocessing/gvc/gvc_test_pt2_allennlp1.0.0.yaml`](resources/yaml/data_preprocessing/gvc/gvc_test_pt2_allennlp1.0.0.yaml) which contains the commented out `python.handwritten_baseline.pipeline.data.processing.masking` pipeline stage for this purpose.

#### Exporting corpus statistics
Corpus statistics such as the number of mentions, the distribution of cluster sizes, the number of coreference links per type, and more (see table 1 in the paper) can be exported automatically.

1. Depending on the corpus to export statistics for, run `export CORPUS=ecbp`, `export CORPUS=fcct`, `export CORPUS=fcc`, or `export CORPUS=gvc`.
2. Then, run:
   ```bash
   cd /cdcr
   source venv_allennlp_1.0.0/bin/activate
   python run_pipeline.py run resources/yaml/device_docker.yaml resources/yaml/data_preprocessing/stats/${CORPUS}.yaml
   ```
2. See `/cdcr/working_dir/${CORPUS}_stats/` for the results.

### Model implementation

#### Feature selection
1. Depending on the corpus to perform feature selection for, run `export CORPUS=ecbp`, `export CORPUS=fcct`, or `export CORPUS=gvc`.
2. Edit `resources/yaml/feature_selection/${CORPUS}.yaml` and fill in `eval_data_path` with the path to the pickle file of the dev split of the respective corpus. 
3. Then, run:
    ```bash
    cd /cdcr
    source venv_allennlp_1.0.0/bin/activate
    python run_feature_model.py feature-selection resources/yaml/device_docker.yaml resources/yaml/feature_selection/${CORPUS}.yaml
    ```
   This will perform [recursive feature elimination](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html) via 6-fold CV over all mention pairs. This process is repeated 7 times with different random seeds, after which the results are aggregated, printed and plotted.
4. The results land in `/cdcr/working_dir/${CORPUS}_feature_selection`. In order to use the selected features in subsequent experiments, the contents of `selected_features.txt` need to be integrated in a YAML config for hyperparameter optimization or training. The exact destination in those YAML files is under `model` → `features` → `selected_features` (see [`/cdcr/resources/yaml/train/ecbp/ecbp_clustering_xgboost.yaml`](resources/yaml/train/ecbp/ecbp_clustering_xgboost.yaml) for an example). 

#### Hyperparameter optimization
Our hyperparameter optimization approach is split into two stages: identifying and optimizing the best mention pair classifier, and (building on top of that) identifying the best hyperparameters for agglomerative clustering.
The hyperparameter sets are sampled automatically with [optuna](https://optuna.org/). See the `sample_classifier_config_with_optuna` method in [`/cdcr/python/handwritten_baseline/pipeline/model/scripts/train_predict_optimize.py`](python/handwritten_baseline/pipeline/model/scripts/train_predict_optimize.py) for the sampling ranges of each hyperparameter. 

1. Depending on the corpus to optimize hyperparameters for, run `export CORPUS=ecbp`, `export CORPUS=fcct`, or `export CORPUS=gvc`.
2. First, to optimize the mention pair classifier:
    1. Pick an ML algorithm: run `export ML=` using the value `lr` for logistic regression, `mlp` for a multi-layer perception, `svm` for a probabilistic SVC classifier or `xgboost` for an [XGBoost](https://xgboost.readthedocs.io/en/latest/) tree boosting classifier.
    2. Edit `/cdcr/resources/yaml/hyperopt/${CORPUS}/${CORPUS}_classifier_${ML}.yaml`.
        * This YAML file is pre-filled with our best results from the feature selection stage for this corpus.
        * Fill in `train_data_path` and `eval_data_path` with the paths to the pickle files of your preprocessed train and dev splits respectively.
        * Update the settings in the `hyperopt` section to your liking. The optimization uses 6-fold cross-validation and runs parallelized. We therefore recommend to choose `cv_num_repeats` so that `6 * cv_num_repeats % max_cores == 0` to minimize the number of idling CPU cores during the optimization. 
    3. Run
       ```bash
       cd /cdcr
       source venv_allennlp_1.0.0/bin/activate
       python run_feature_model.py hyperopt resources/yaml/device_docker.yaml resources/yaml/hyperopt/${CORPUS}/${CORPUS}_classifier_${ML}.yaml
       ```
       A plot with the optimization progress will be generated every 10 trials in `/cdcr/working_dir/${CORPUS}_hyperopt_classifier_${ML}`.
    4. See `/cdcr/working_dir/${CORPUS}_hyperopt_classifier_${ML}` for the results. To use the optimal hyperparameters in subsequent experiments, the contents of `best_model_config.yaml` need to be integrated in a YAML config for hyperparameter optimization or training inside the `model` section (see [`/cdcr/resources/yaml/train/ecbp/ecbp_clustering_xgboost.yaml`](resources/yaml/train/ecbp/ecbp_clustering_xgboost.yaml) for an example). 
3. To optimize the agglomerative clustering step:
    1. Edit `/cdcr/resources/yaml/hyperopt/${CORPUS}/${CORPUS}_clustering_${ML}.yaml`. Due to XGBoost performing best in our experiments, we provide a pre-filled version of this file only for `ML=xgboost`. Fill in `train_data_path` and `eval_data_path`.
    2. Run
       ```bash
       cd /cdcr
       source venv_allennlp_1.0.0/bin/activate
       python run_feature_model.py hyperopt resources/yaml/device_docker.yaml resources/yaml/hyperopt/${CORPUS}/${CORPUS}_clustering_${ML}.yaml
       ```
    3. See `/cdcr/working_dir/${CORPUS}_hyperopt_clustering_${ML}` for the results, particularly the `best_model_config.yaml` file which contains the best identified hyperparameters as in the previous step.

#### Document preclustering at test time
The system can be evaluated with predefined document clusters at test time.
This can be achieved by specifying a path to a pickle file for the `hard_document_clusters_file` option in any of the YAML configurations inside [`/cdcr/resources/yaml/evaluate/`](resources/yaml/evaluate).
The pickle file is expected to contain a list of lists of document identifiers representing the clustering. One such file is produced by the data preprocessing pipeline (named `*_gold_transitive_closure_doc_clustering.pkl `) which contains the gold clustering of the split as defined by the transitive closure over mentions.

#### Separate training/evaluation of the mention pair classifier
The mention pair classifier component of the system can be evaluated separately (see sections 6.3.2 and 6.3.3 in the paper):

1. Perform steps 1 and 2i of [the hyperparameter optimization instructions](#hyperparameter-optimization).
2. Edit `/cdcr/resources/yaml/train/${CORPUS}/${CORPUS}_classifier_${ML}.yaml`:
    * Due to XGBoost performing best in our experiments, we provide a pre-filled version of this file only for `ML=xgboost`. When using xgboost as a classifier, it is possible to enable the `analyze_feature_importance` option which will produce the `feature_importances.csv` and `feature_importances_aggregated.txt` files reporting the gain from each feature. 
    * Fill in `train_data_path` with the paths to the pickle files of your preprocessed train split.
3. Run
   ```bash
   cd /cdcr
   source venv_allennlp_1.0.0/bin/activate
   python run_feature_model.py train resources/yaml/device_docker.yaml resources/yaml/train/${CORPUS}/${CORPUS}_classifier_${ML}.yaml
   ```
   The trained models appear in `/cdcr/working_dir/${CORPUS}_training_classifier_${ML}` in a `serialized_models` folder.
4. Edit `/cdcr/resources/yaml/evaluate/${CORPUS}/${CORPUS}_classifier.yaml`. Fill in `eval_data_path` with the path to the pickle file of your preprocessed test split.  
5. Run (with `/path/to/serialized_models` set to the path with models from step 3):
   ```bash
   python run_feature_model.py evaluate /path/to/serialized_models resources/yaml/device_docker.yaml resources/yaml/evaluate/${CORPUS}/${CORPUS}_classifier.yaml
   ```
   The following files are generated:
   ```
   /cdcr/working_dir/${CORPUS}_evaluate_classifier/
   └── bec23be8
       └── 2020-11-12_12-37-42
           ├── 0
           │   └── outcomes.pkl                                              # serialized predictions of model with random seed 0
           ├── ...
           ├── detailed_metrics
           │   ├── p_r_f1_average_over_runs_ignoring_link_types.txt          # Mean P, R, F1 over all runs, ignoring CDCR link types
           │   ├── p_r_f1_average_over_runs_for_each_link_type.txt           # Mean P, R, F1 over all runs, but for each CDCR link type separately
           │   ├── mean_absolute_confusion_matrix_quadrants_over_runs.txt    # Mean absolute number of TP, FP, ... predictions for each CDCR link type over all runs
           │   └── *.pkl                                                     # serialized pandas DataFrames of each of the pretty-printed tables
           ├── ...
           ├── metrics_aggregated_pretty.txt                                 # same content as p_r_f1_average_over_runs_for_each_link_type.txt
           └── prediction_examples
               ├── prediction_examples.csv                                   # contains as many predicted mention pairs per confusion matrix quadrant and each link type as specified by the 'num_samples_per_quadrant' setting, in machine-readable form
               ├── cross-subtopic pairs                                      # contains pretty-printed cross-subtopic link predictions
               │   ├── TXT_FN_prediction_examples.txt                        # pretty-printed false negatives mention pairs, with document context and action mention spans >>>emphasized<<<
               │   ├── TXT_FP_prediction_examples.txt                        # false positives, and so on
               │   ├── TXT_TN_prediction_examples.txt
               │   ├── TXT_TP_prediction_examples.txt
               │   └── TEX_*.tex                                             # all the above in LaTeX form
               ├── cross-topic pairs                                         # pretty-printed cross-topic link predictions, and so on
               │   ├── ...
               ├── within-document pairs
               │   ├── ...
               └── within-subtopic pairs
                   └── ...
   ```
   The content of `metrics_aggregated_pretty.txt` is logged to stdout upon completion of the script.

#### Running baseline experiments
To reproduce the `lemma`, `lemma-delta` and `lemma-time` baseline experiments (reported in table 5 in the paper):
1. Depending on which corpus to run baselines for, run `export CORPUS=ecbp`, `export CORPUS=fcct`, or `export CORPUS=gvc`.
2. Edit `/cdcr/resources/yaml/lemma_baselines/${CORPUS}.yaml` and fill in `train_data_path` and `eval_data_path` with the paths to the pickle files of your preprocessed train and test splits respectively.
3. Run
   ```bash
   cd /cdcr
   source venv_allennlp_1.0.0/bin/activate
   python run_feature_model.py lemma-baselines resources/yaml/device_docker.yaml resources/yaml/lemma_baselines/${CORPUS}.yaml
   ```
   This triggers optimization of the `lemma-delta` and `lemma-time` baselines, followed by prediction and evaluation with coreference resolution metrics.
4. The following files are generated:
   ```
   /cdcr/working_dir/${CORPUS}_lemma_baselines
   └── b9d92fee
       └── 2020-11-12_15-02-52
           ├── results
           │   ├── lemma
           │   │   ├── cross_doc                       # evaluation scenario where all documents are merged into a single meta-document in the gold and system CoNLL files. This is what we report in the paper.
           │   │   │   ├── eval_lemma_cross_doc.txt    # output from the scoring script
           │   │   │   ├── gold.conll                  # gold annotations - the naming scheme for mentions is `metadoc_{document id}_{mention id}`
           │   │   │   └── system.conll                # predictions
           │   │   └── within_doc                      # evaluation scenario where documents are scored in isolation, i.e. only within-document links are scored and cross-document links are left out entirely.
           │   │       └── ...
           │   ├── lemma-delta
           │   │   └── ...
           │   ├── lemma-time
           │   │   └── ...
           │   └── lemma-wd                            # not reported in the paper; only performs within-document coreference resolution (see Upadhyay 2016 et al. https://www.aclweb.org/anthology/C16-1183/)
           │       └── ...
           ├── tfidf                                   # optimization progress reports of the lemma-tfidf optimization
           └── time                                    # reports of the lemma-time optimization
   ```

## Troubleshooting
* This implementation caches the results of several operations to disk to speed up subsequent executions. For example, the corpus preprocessing pipeline caches queries to CoreNLP, DBpedia, the SRL system and more. The training/prediction code caches mention pair features which considerably speeds up hyperparameter optimization. In case of problems (or when modifying these components), it may be necessary to clear the affected caches. Do this by manually removing the corresponding cache folders inside `/cdcr/working_dir/global`. 
* If you encounter OOM issues during hyperparameter optimization, try reducing `max_cores` in `device_docker.yaml`.
* Please open a github issue in case of other problems.

## References
* ECB+ corpus: http://www.newsreader-project.eu/results/data/the-ecb-corpus/
* Gun Violence Corpus (GVC): https://github.com/cltl/GunViolenceCorpus
* CoVal: A coreference evaluation tool for the CoNLL and ARRAU datasets: https://github.com/ns-moosavi/coval
