from logging import Logger
from pathlib import Path
from typing import List, Tuple, Any, Optional, Dict

import numpy as np
import optuna
import pandas as pd
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.pipeline import Pipeline

from python import DOCUMENT_ID, EVENT_ID, SENTENCE_IDX, TOKEN_IDX_FROM, TOKEN_IDX_TO
from python.handwritten_baseline.pipeline.model.scripts.evaluation_utils import run_conll_evaluation
from python.handwritten_baseline import LEMMA
from python.handwritten_baseline.pipeline.model.data_prep.pipeline_data_input import get_X_and_y_for_pipeline, \
    create_gold_clustering
from python.handwritten_baseline.pipeline.model.scripts.baselines.doc_clustering import create_doc_clustering_pipeline
from python.handwritten_baseline.pipeline.model.scripts.scoring import obtain_final_coref_metrics
from python.handwritten_baseline.pipeline.model.scripts.train_predict_optimize import load_data
from python.pipeline import RUN_WORKING_DIR
from python.util.optuna import PlotCallback

LEMMA_BASELINE = "lemma"
LEMMA_WD_BASELINE = "lemma-wd"
LEMMA_DELTA_BASELINE = "lemma-delta"
LEMMA_TIME_BASELINE = "lemma-time"

def predict_baseline(logger: Logger,
                     X: List[Tuple],
                     y: List[Any],
                     which: str,
                     doc_clustering_pipeline: Optional[Pipeline] = None) -> Tuple[pd.Series, pd.Series]:
    """

    :param logger:
    :param X:
    :param y:
    :param which: which baseline to run
    :param doc_clustering_pipeline: only used for lemma-delta
    :return: tuple of gold clustering and predicted cluster
    """
    assert len(X) == 1 and len(y) == 1

    dataset, documents, mentions = X[0]
    mentions_to_gold_events = y[0]

    # if a document clustering pipeline is given, apply it, otherwise put all documents in the same cluster
    if which in [LEMMA_DELTA_BASELINE, LEMMA_TIME_BASELINE]:
        if doc_clustering_pipeline is None:
            raise ValueError
        doc_clustering = doc_clustering_pipeline.predict(X)
        logger.info(f"Document clustering created {len(doc_clustering[EVENT_ID].unique())} cluster(s).")
    else:
        docs = documents[DOCUMENT_ID].values
        doc_clustering = pd.DataFrame({DOCUMENT_ID: docs, EVENT_ID: pd.Series(np.zeros_like(docs, dtype=np.int))})

    # obtain lemma for each action phrase, creating those in a per-document fashion is easiest
    mention_lemmas_in_docs = []
    for doc_id, df in mentions.groupby(DOCUMENT_ID):
        lemmas_in_doc = dataset.tokens.loc[(doc_id), LEMMA]

        mention_lemmas_in_doc = df.apply(lambda row: " ".join(lemmas_in_doc.loc[row[SENTENCE_IDX], slice(row[TOKEN_IDX_FROM], row[TOKEN_IDX_TO] - 1)]), axis=1)

        # for the lemma-wd baseline, make sure we have unique mention lemmas in each document by prepending the doc id
        if which == LEMMA_WD_BASELINE:
            mention_lemmas_in_doc = doc_id + "__" + mention_lemmas_in_doc

        mention_lemmas_in_docs.append(mention_lemmas_in_doc)
    mention_lemmas_in_docs = pd.concat(mention_lemmas_in_docs)

    # confine mention lemmas to document clusters by prepending the cluster id to each lemma
    for cluster_id, doc_ids in doc_clustering.groupby(EVENT_ID)[DOCUMENT_ID]:
        # remove documents which do not have mentions
        doc_ids = doc_ids.loc[doc_ids.isin(mention_lemmas_in_docs.index.get_level_values(DOCUMENT_ID))]
        mention_lemmas_in_docs.loc[doc_ids] = str(cluster_id) + "__" + mention_lemmas_in_docs.loc[doc_ids]

    # don't let the name of the method fool you into thinking there is gold data involved here, it just does exactly
    # the thing we want to do here
    system_clusters = create_gold_clustering(mention_lemmas_in_docs)
    gold_clusters = create_gold_clustering(mentions_to_gold_events)

    # Our clustering approach cannot produce multiple clusters for one mention. We still need to map the integer
    # cluster identifiers in lists.
    gold_clusters = gold_clusters.map(lambda v: [v])
    system_clusters = system_clusters.map(lambda v: [v])

    return gold_clusters, system_clusters


def optimize_thresh(train_X,
                    train_y,
                    signal: str,
                    min_thresh: float,
                    max_thresh: float,
                    logger: Logger,
                    serialization_dir: Path,
                    n_trials: int = 50):
    # Optimize the clustering thresh for tunable baselines: We fit a doc clustering pipeline on the training split, then
    # we predict document clusters on train, use those to run lemma delta and evaluate with LEA F1 against the gold
    # event coref clusters. Barhom et al. optimized the document clustering performance itself, we optimize document
    # clustering + coref which is more faithful to the actual task and should provide fairer, more comparable results
    # across multiple datasets.
    def objective(trial: Trial):
        delta = trial.suggest_float("delta", min_thresh, max_thresh)
        pipeline = create_doc_clustering_pipeline(delta, signal)
        pipeline.fit(train_X)

        gold_clusters, system_clusters = predict_baseline(logger, train_X, train_y,
                                                          which=LEMMA_DELTA_BASELINE,
                                                          doc_clustering_pipeline=pipeline)

        metrics = run_conll_evaluation(gold_clusters, system_clusters, single_meta_document=True, metrics="lea")
        lea_f1 = metrics.loc["lea", "f1"]
        return lea_f1

    sampler = TPESampler(seed=1)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    study.optimize(objective, n_trials=n_trials, callbacks=[PlotCallback(serialization_dir=serialization_dir / signal / "plots")])
    best_delta = study.best_trial.params["delta"]
    return best_delta


def run_baselines(config_data: Dict,
                  config_baselines: List,
                  config_global: Dict,
                  logger: Logger) -> None:
    """
    Runs lemma baseline, lemma-delta baseline on the given dataset and writes scores to `serialization_dir`.
    The delta hyperparameter for lemma-delta is optimized by training on train and evaluating on dev.
    :param config_data
    :param config_baselines
    :param config_global
    :param logger
    """

    serialization_dir = config_global[RUN_WORKING_DIR]

    # load datasets
    def load_split(p):
        data = load_data(p)
        X, y = get_X_and_y_for_pipeline(logger,
                                        data,
                                        doc_partitioning=None,
                                        oracle_mention_pair_generation=False)
        return X, y

    train_X, train_y = load_split(config_data["train_data_path"])
    eval_X, eval_y = load_split(config_data["eval_data_path"])

    lemma_delta_best_thresh = None
    if LEMMA_DELTA_BASELINE in config_baselines:
        logger.info("Starting lemma-delta threshold optimization.")
        lemma_delta_best_thresh = optimize_thresh(train_X, train_y, "tfidf", 0.0, 1.0, logger, serialization_dir)
        logger.info(f"Best thresh for lemma-delta: {lemma_delta_best_thresh}")

    lemma_time_best_thresh = None
    if LEMMA_TIME_BASELINE in config_baselines:
        logger.info("Starting lemma-time threshold optimization.")
        lemma_time_best_thresh = optimize_thresh(train_X, train_y, "time", 0.0, 10 * 365 * 24, logger, serialization_dir)
        logger.info(f"Best thresh for lemma-time: {lemma_time_best_thresh}")

    evaluation_results_dir = serialization_dir / "results"
    evaluation_results_dir.mkdir(exist_ok=True)
    for baseline_name in config_baselines:
        logger.info(f"Predicting for {baseline_name}")
        if baseline_name == LEMMA_DELTA_BASELINE:
            # fit tfidf on eval
            pipeline = create_doc_clustering_pipeline(lemma_delta_best_thresh, signal="tfidf")
            pipeline.fit(eval_X)
        elif baseline_name == LEMMA_TIME_BASELINE:
            pipeline = create_doc_clustering_pipeline(lemma_time_best_thresh, signal="time")
        else:
            pipeline = None

        gold_clusters, system_clusters = predict_baseline(logger, eval_X, eval_y,
                                                          which=baseline_name,
                                                          doc_clustering_pipeline=pipeline)

        dir_of_baseline = evaluation_results_dir / baseline_name
        dir_of_baseline.mkdir(exist_ok=True)

        logger.info(f"Computing metrics for {baseline_name}")
        obtain_final_coref_metrics(gold_clusters,
                                   system_clusters,
                                   f"eval_{baseline_name}",
                                   "all",
                                   serialization_dir=dir_of_baseline)