import copy
import json
import pickle
import pprint
from logging import Logger
from pathlib import Path
from typing import Dict, Optional, List, Union, Tuple

import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from joblib import dump, delayed, Parallel, load
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RepeatedKFold, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from tabulate import tabulate

from python.handwritten_baseline.pipeline.model.classifier_clustering.pairwise_classifier_wrapper import \
    PredictOnTransformClassifierWrapper
from python.handwritten_baseline.pipeline.model.data_prep.pipeline_data_input import get_X_and_y_for_pipeline
from python.handwritten_baseline.pipeline.model.feature_extr import LEMMA_EXTR, TFIDF_EXTR, TIME_EXTR, LOCATION_EXTR, \
    SENTENCE_EMBEDDING_EXTR, ACTION_PHRASE_EMBEDDING_EXTR, WIKIDATA_EMBEDDING_EXTR
from python.handwritten_baseline.pipeline.model.scripts import SVC_HUBER, LOGISTIC_REGRESSION, _TYPE, _KWARGS, XGBOOST, \
    _FIT_PARAMS, MLP
from python.handwritten_baseline.pipeline.model.scripts.feature_importance import get_feature_names_from_pipeline, \
    analyze_feature_importance
from python.handwritten_baseline.pipeline.model.scripts.pipeline_instantiation import instantiate_pipeline, \
    CLUSTERING_PIPELINE_STEP_NAME, CLASSIFIER_PIPELINE_STEP_NAME
from python.handwritten_baseline.pipeline.model.scripts.prediction_analysis import perform_prediction_analysis
from python.handwritten_baseline.pipeline.model.scripts.scoring import CrossDocCorefScoring, MentionPairScoring
from python.pipeline import RUN_WORKING_DIR, MAX_CORES
from python.util.config import write_config
from python.util.optuna import EarlyStoppingCallback, PlotCallback
from python.util.util import get_dict_hash


def load_data(path):
    # load preprocessed dataset from file
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def sample_classifier_config_with_optuna(trial: Trial, classifier_name: str) -> Dict:
    """
    Uses optuna to sample a config with classifier hyperparameters.
    :param trial: Optuna trial
    :param classifier_name: The classifier to use (and sample hyperparameters for). Testing them separately seems to
                        make more sense to me.
    :return: classifier config
    """
    if classifier_name in [SVC_HUBER, LOGISTIC_REGRESSION]:
        if classifier_name == SVC_HUBER:
            # modified_huber results in a quadratically smoothed SVM with gamma = 2
            loss = "modified_huber"
        elif classifier_name == LOGISTIC_REGRESSION:
            loss = "log"
        else:
            raise ValueError

        # alpha range follows the suggestions of the sklearn documentation
        classifier_config = {_TYPE: "SGDClassifier",
                             _KWARGS: {"loss": loss,
                                       "alpha": trial.suggest_loguniform("alpha", 1e-7, 1e-1),
                                       "max_iter": 1000,
                                       "early_stopping": True,
                                       "validation_fraction": 0.1,
                                       "n_iter_no_change": 5}}
    elif classifier_name == XGBOOST:
        classifier_config = {_TYPE: "ConvenientXGBClassifier",
                             _KWARGS: {"n_jobs": 1,
                                       "n_estimators": 1000,  # we use early stopping, so this is the maximum
                                       "learning_rate": trial.suggest_loguniform("learning_rate", 1e-4, 1e0),
                                       # learning rate
                                       "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
                                       # min required instance weight at a child
                                       "max_depth": trial.suggest_int("max_depth", 3, 12),  # max tree depth
                                       "gamma": trial.suggest_loguniform("gamma", 1e-3, 1e0),
                                       # Minimum loss reduction required to make a further partition on a leaf node of the tree.
                                       "max_delta_step": trial.suggest_loguniform("max_delta_step", 1e-3, 1e2),
                                       # Maximum delta step we allow each leaf output to be. Reported to help with imbalanced data.
                                       "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                                       "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                                       "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
                                       # recommended to use for imbalanced datasets (which we definitely have)
                                       "scale_pos_weight": trial.suggest_loguniform("scale_pos_weight", 1.0, 10),
                                       "objective": "binary:logistic",
                                       "eval_metric": "logloss",
                                       },
                             _FIT_PARAMS: {"early_stopping_rounds": 5,
                                           "eval_metric": "logloss",
                                           "validation_fraction": 0.1,
                                           "verbose": False}}
    elif classifier_name == MLP:
        num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 2)
        last_hidden_layer_size = trial.suggest_int("last_hidden_layer_size", 5, 50)
        hidden_layer_sizes = [2 ** (num_hidden_layers - i - 1) * last_hidden_layer_size for i in
                              range(num_hidden_layers)]

        classifier_config = {_TYPE: "MLPClassifier",
                             _KWARGS: {"hidden_layer_sizes": tuple(hidden_layer_sizes),
                                       "activation": "relu",
                                       "solver": "adam",
                                       "learning_rate_init": trial.suggest_loguniform("learning_rate_init", 1e-4, 1e-1),
                                       "max_iter": 1000,
                                       "shuffle": True,
                                       "early_stopping": True,
                                       "n_iter_no_change": 5,
                                       "validation_fraction": 0.1}}
    else:
        raise ValueError
    return classifier_config


def sample_clustering_config_with_optuna(trial: Trial) -> Dict:
    """
    Uses optuna to sample a config dictionary with clustering parameters.
    :param trial: optuna trial
    :return: config dictionary
    """
    cluster_criterion = trial.suggest_categorical("cluster_criterion", ['inconsistent', 'distance', 'maxclust'])
    cluster_depth = 0 if not cluster_criterion == 'inconsistent' else trial.suggest_int("cluster_depth", low=1, high=10)
    clustering_config = {"threshold": trial.suggest_uniform("threshold", 0, 1),
                         "linkage_method": trial.suggest_categorical("linkage_method", ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']),
                         "cluster_criterion": cluster_criterion,
                         "cluster_depth": cluster_depth}
    return clustering_config


def get_feature_extractors_config_with_all_and_defaults() -> Dict:
    """
    Returns config section for all feature extractors with default values.
    :return:
    """
    return {
        LEMMA_EXTR: {},
        TFIDF_EXTR: {},
        TIME_EXTR: {},
        LOCATION_EXTR: {},
        SENTENCE_EMBEDDING_EXTR: {},
        ACTION_PHRASE_EMBEDDING_EXTR: {},
        WIKIDATA_EMBEDDING_EXTR: {}
    }


def optimize_hyperparameters(config_data: Dict,
                             config_model: Dict,
                             config_hyperopt: Dict,
                             config_global: Dict,
                             logger: Logger):
    """
    To be used for hyperparameter optimization of the mention pair classifier and agglomerative clustering.
    :param config_data:
    :param config_model:
    :param config_hyperopt:
    :param config_global:
    :param logger:
    :return:
    """
    # During the hyperparameter optimization, use a fixed random seed for the Optuna sampling, CV splits and classifier.
    optimization_random_seed = 0

    # If False, hyperparameters for mention pair classification are optimized. If True, hyperparameters for clustering
    # are optimized. The latter case needs a full classifier configuration, see below.
    with_clustering = config_hyperopt["with_clustering"]
    classifier = config_model["classifier"] # type: Union[str, Dict]

    # ------------- validate parameters ---------------
    if not with_clustering and (classifier is None or type(classifier) is dict):
        raise ValueError("To optimize the mention pair classifier, the 'classifier' config parameter must be the name of the classifier to optimize.")

    if with_clustering and (type(classifier) is str or not classifier):
        raise ValueError("To optimize the clustering step, the 'classifier' config parameter must be a complete classifier configuration in the form of a dictionary.")

    # ------------- create base config to more or less use in each optimization step ------------
    extractors = config_model["features"].get("extractors", None)
    if extractors is None:
        extractors = get_feature_extractors_config_with_all_and_defaults()

    # Pass this to filter extracted features so that only those from preliminary feature selection are used.
    # None means "use all features", an empty list means no features at all!
    selected_features = config_model["features"].get("selected_features", None)     # type: Optional[List]

    pairs_config = config_data["pairs"]

    base_config = {"random_seed": optimization_random_seed,
                   "features": {
                       "extractors": extractors,
                       "selected_features": selected_features
                   },
                   "pairs": pairs_config
                   }

    # ------------- get going with optimization now ---------------

    serialization_dir = config_global[RUN_WORKING_DIR]

    train_data = load_data(config_data["train_data_path"])
    doc_partitioning = config_data["doc_partitioning"]
    oracle_mention_pair_generation = config_data["oracle_mention_pair_generation"]

    train_X, train_y = get_X_and_y_for_pipeline(logger,
                                                train_data,
                                                doc_partitioning=doc_partitioning,
                                                oracle_mention_pair_generation=oracle_mention_pair_generation)

    # for cross-validation, make 6 splits at most and fall back to leave-one-out (here: one instance = one partition)
    # if there are few partitions
    cv_num_splits = min(6, len(train_X))
    cv_num_repeats = config_hyperopt["cv_num_repeats"]
    cv_n_jobs = config_global[MAX_CORES]
    if cv_n_jobs > 1 and ((cv_num_splits * cv_num_repeats) % cv_n_jobs) != 0:
        logger.warning(f"Inefficient cross-validation parameter choices, expect idling CPUs ({cv_num_splits} folds * {cv_num_repeats} repeats % {cv_n_jobs} CPUs != 0)")

    def objective(trial: Trial):
        # config dictionaries are modified during instantiation, so we need to deepcopy the originals to not lose them
        config = copy.deepcopy(base_config)
        if with_clustering:
            assert type(classifier) is not str
            config["classifier"] = copy.deepcopy(classifier)
            config["clustering"] = sample_clustering_config_with_optuna(trial)
        else:
            assert type(classifier) is str
            config["classifier"] = sample_classifier_config_with_optuna(trial, classifier)

        # store the config in the trial so that we can retrieve it later and use it to instantiate the best model -
        # don't ask me why it needs to be stored as a string, using the dict object did not work
        trial.set_user_attr("config", json.dumps(config))

        # instantiate feature pipeline and classifier, transform the features
        pipeline, scoring = instantiate_pipeline(logger,
                                                 config,
                                                 with_clustering=with_clustering,
                                                 use_caching=True,
                                                 scorer_should_return_single_scalar=True,
                                                 serialization_dir=serialization_dir / "pipeline" / f"trial_{trial.number:03}")

        cv = RepeatedKFold(n_splits=cv_num_splits,
                           n_repeats=cv_num_repeats,
                           random_state=optimization_random_seed)

        f1_scores_cv = cross_val_score(estimator=pipeline,
                                       X=train_X,
                                       y=train_y,
                                       n_jobs=cv_n_jobs,
                                       cv=cv,
                                       scoring=scoring,
                                       verbose=0)
        mean_f1 = f1_scores_cv.mean()
        return mean_f1

    logger.info("Starting optimization.")
    callbacks = []
    if "early_stopping" in config_hyperopt:
        callbacks.append(EarlyStoppingCallback(logger, **config_hyperopt["early_stopping"]))
    callbacks.append(PlotCallback(serialization_dir=serialization_dir / "plots"))

    sampler = TPESampler(seed=optimization_random_seed)
    study = optuna.create_study(sampler=sampler, direction="maximize")

    optuna_timeout_seconds = pd.to_timedelta(config_hyperopt["timeout"]).total_seconds()
    optuna_n_trials = config_hyperopt["n_trials"]
    study.optimize(objective,
                   n_trials=optuna_n_trials,
                   timeout=optuna_timeout_seconds,
                   callbacks=callbacks)
    best_trial = study.best_trial
    best_config = json.loads(best_trial.user_attrs["config"])

    logger.info("Best trial: " + repr(best_trial))
    logger.info("Best config:\n" + pprint.pformat(best_config))

    # write best config to file
    best_config_file = serialization_dir / "best_model_config.yaml"
    write_config(best_config, best_config_file)


def train(config_data: Dict,
          config_model: Dict,
          config_training: Dict,
          config_global: Dict,
          logger: Logger) -> None:
    """
    Trains n classifier+clustering pipelines with a given configuration.
    :param config_data:
    :param config_model:
    :param config_training:
    :param config_global:
    :param logger:
    :return:
    """
    serialization_dir = config_global[RUN_WORKING_DIR]

    num_models_to_train = config_training["num_models_to_train"]
    with_clustering = config_training["with_clustering"]

    train_data = load_data(config_data["train_data_path"])
    doc_partitioning = config_data["doc_partitioning"]
    oracle_mention_pair_generation = config_data["oracle_mention_pair_generation"]
    train_X, train_y = get_X_and_y_for_pipeline(logger,
                                                train_data,
                                                doc_partitioning=doc_partitioning,
                                                oracle_mention_pair_generation=oracle_mention_pair_generation)

    base_pipeline_config = {**config_model,
                            "pairs": config_data["pairs"]}
    if base_pipeline_config["features"]["extractors"] is None:
        base_pipeline_config["features"]["extractors"] = get_feature_extractors_config_with_all_and_defaults()

    def fit_save_and_report(random_seed: int) -> Pipeline:
        pipeline_config = copy.deepcopy(base_pipeline_config)
        pipeline_config["random_seed"] = random_seed
        pipeline, scoring = instantiate_pipeline(logger,
                                                 pipeline_config,
                                                 with_clustering=with_clustering,
                                                 scorer_should_return_single_scalar=False,
                                                 serialization_dir=serialization_dir / "pipeline" / f"seed_{random_seed:03}")
        pipeline.fit(X=train_X, y=train_y)
        return pipeline

    # train pipelines in parallel
    logger.info(f"Training {num_models_to_train} separate models...")
    jobs = [delayed(fit_save_and_report)(random_seed) for random_seed in range(num_models_to_train)]
    pipelines = Parallel(n_jobs=config_global[MAX_CORES])(jobs)

    if config_training["analyze_feature_importance"]:
        logger.info("Analyzing feature importance")
        analyze_feature_importance(pipelines, serialization_dir, logger)

    logger.info("Saving pipelines to disk")
    model_dir = serialization_dir / "serialized_models"
    model_dir.mkdir(exist_ok=True)
    for i, p in enumerate(pipelines):
        dump(p, model_dir / f"{i}.pipeline.joblib")


def evaluate(model_serialization_dir: Path,
             config_data: Dict,
             config_evaluate: Dict,
             config_global: Dict,
             logger: Logger) -> pd.DataFrame:
    """
    Predicts and evaluates
    :param model_serialization_dir: path to the directory containing serialized models and scorers
    :param config_data:
    :param config_evaluate:
    :param config_global:
    :param logger:
    :return: metrics Dataframe
    """
    serialization_dir = Path(config_global[RUN_WORKING_DIR])

    logger.info("Finding and loading model pipelines from disk.")
    pipelines = {}  # type: Dict[int, Pipeline]
    for p in model_serialization_dir.iterdir():
        i = int(p.stem.split(".")[0])
        if "".join(p.suffixes) == ".pipeline.joblib":
            pipelines[i] = load(p)

    # find out if we are dealing with mention pair classification or clustering pipelines
    last_pipeline_step_names = {p.steps[-1][0] for p in pipelines.values()}
    if len(last_pipeline_step_names) > 1:
        raise ValueError("All pipelines must be of the same type (mention pair classification or clustering)")
    last_pipeline_step_name = list(last_pipeline_step_names)[0]

    # prepare scorers
    if last_pipeline_step_name == CLASSIFIER_PIPELINE_STEP_NAME:
        is_clustering_pipeline = False

        # collect mention pair scorer parameters
        if not "pairs" in config_data:
            raise ValueError("Scoring mention pairs requires a 'pairs' config.")
        config_pairs = config_data["pairs"]
        mpg_prediction_config = config_pairs.pop("mpg_prediction")
        if mpg_prediction_config is not None:
            logger.warning("'mpg_prediction' was specified for a mention pair scoring scenario. Depending on those parameters, evaluation results are not representative. I hope you know what you're doing.")
    elif last_pipeline_step_name == CLUSTERING_PIPELINE_STEP_NAME:
        is_clustering_pipeline = True

        # if present, inject hard document clusters into the last pipeline stage (the clustering stage)
        hard_document_clusters_file = config_evaluate["hard_document_clusters_file"]
        if hard_document_clusters_file is not None:
            hard_document_clusters_file = Path(hard_document_clusters_file)
            assert hard_document_clusters_file.exists() and hard_document_clusters_file.is_file()

            with hard_document_clusters_file.open("rb") as f:
                hard_document_clusters = pickle.load(f)

            # the format in the pickle file is topic_subtopic-part-1_..._subtopic-part-n_doc-id to be used with the Barhom et al. system, so we split on underscores and pick the last value to obtain the document id
            hard_document_clusters = [{doc_id.split("_")[-1] for doc_id in cluster} for cluster in hard_document_clusters]

            logger.info(f"Using hard document clustering ({len(hard_document_clusters)} clusters given).")
            for p in pipelines.values():
                p.steps[-1][1].set_params(hard_document_clusters=hard_document_clusters)
    else:
        raise ValueError("Could not identify last pipeline step.")

    # load and prepare data
    eval_data = load_data(config_data["eval_data_path"])
    doc_partitioning = config_data["doc_partitioning"]
    oracle_mention_pair_generation = config_data["oracle_mention_pair_generation"]
    eval_X, eval_y = get_X_and_y_for_pipeline(logger,
                                              eval_data,
                                              doc_partitioning=doc_partitioning,
                                              oracle_mention_pair_generation=oracle_mention_pair_generation)

    def predict_and_evaluate(i, pipeline):
        # write scoring outputs into separate folder for each model
        i_serialization_dir = serialization_dir / str(i)
        i_serialization_dir.mkdir(exist_ok=True)

        # instantiate scorer which fits the pipeline
        if is_clustering_pipeline:
            scorer = CrossDocCorefScoring(metrics="all", serialization_dir=i_serialization_dir)
        else:
            scorer = MentionPairScoring(mpg_prediction_config,
                                        serialization_dir=i_serialization_dir)
        metrics, outcomes = scorer(pipeline, eval_X, eval_y)
        metrics["model"] = i

        return metrics, outcomes

    # predict in parallel
    logger.info(f"Predicting/evaluating {len(pipelines)} separate models...")
    jobs = [delayed(predict_and_evaluate)(i, pipeline) for i, pipeline in pipelines.items()]
    metrics_and_outcomes = Parallel(n_jobs=config_global[MAX_CORES])(jobs)

    metrics, outcomes = list(zip(*metrics_and_outcomes))

    # for classifiers only: detailed prediction analysis for each coref link type and prediction examples
    if not is_clustering_pipeline and config_evaluate["perform_prediction_analysis"]:
        logger.info(f"Performing prediction analysis")
        num_samples_per_quadrant = config_evaluate["num_samples_per_quadrant"]
        perform_prediction_analysis(dataset=eval_data,
                                    outcomes=outcomes,
                                    num_samples_per_quadrant=num_samples_per_quadrant,
                                    serialization_dir=serialization_dir)

    # aggregate metrics: min/max/mean/std
    metrics = pd.concat(metrics)
    if is_clustering_pipeline:
        group_by = ["meta-doc", "metric"]
    else:
        group_by = ["metric"]
    metrics_agg = metrics.groupby(group_by)[["f1", "precision", "recall"]].describe(percentiles=[])
    metrics_agg.drop(columns=["count", "50%"], level=1, inplace=True)

    # write metrics to disk
    metrics.to_csv(serialization_dir / "metrics_unaggregated.csv", index=True)
    metrics_agg.to_csv(serialization_dir / "metrics_aggregated.csv", index=True)
    metrics_agg_str = tabulate(metrics_agg, headers="keys")
    with (serialization_dir / "metrics_aggregated_pretty.txt").open("w") as f:
        f.write(metrics_agg_str)
    logger.info("\n" + metrics_agg_str)

    return metrics_agg


def feature_selection(config_data: Dict,
                      config_global: Dict,
                      logger: Logger):
    """
    Runs feature selection on the EVALUATION split.
    Uses 10 runs of 5-fold cross-validation for recursive feature elimination with a Random Forest mention classifier to
    find the most useful features.
    :param config_data:
    :param config_global:
    :param logger:
    :return:
    """
    serialization_dir = config_global[RUN_WORKING_DIR]

    eval_data_path = config_data["eval_data_path"]
    oracle_mention_pair_generation = config_data["oracle_mention_pair_generation"]

    data = load_data(eval_data_path)
    X, y = get_X_and_y_for_pipeline(logger,
                                    data,
                                    doc_partitioning=None,
                                    oracle_mention_pair_generation=oracle_mention_pair_generation)

    config_base = {
              "classifier": {_TYPE: "RandomForest",
                             _KWARGS: {"n_estimators": 100}},
              "features": {
                  "extractors": get_feature_extractors_config_with_all_and_defaults(),
                  "selected_features": None
              },
              "pairs": config_data["pairs"]
              }

    def run_rfecv_iteration(random_seed: int,
                            n_splits: int = 6) -> Tuple[List[str], np.array, np.array]:
        # RFECV needs X to be an matrix-like of shape (n_samples, n_features). This means we cannot use our pipeline as is,
        # because our X's are not matrix-like. So we run our pipeline up to the point where we input the feature matrix +
        # labels into the mention pair classifier, and feed that to RFECV. To do that, we need to chop up the pipeline.
        config = copy.deepcopy(config_base)
        config["random_seed"] = random_seed

        pipeline, scoring = instantiate_pipeline(logger,
                                                 config,
                                                 with_clustering=False,
                                                 scorer_should_return_single_scalar=True,
                                                 serialization_dir=serialization_dir / "pipeline")

        # remove the classifier at the end of the pipeline
        classifier_wrapper = pipeline.steps.pop(-1)[1]  # type: PredictOnTransformClassifierWrapper
        assert type(classifier_wrapper) is PredictOnTransformClassifierWrapper
        random_forest_clf = classifier_wrapper.classifier_

        # obtain feature matrix and labels
        conflated_X = pipeline.fit_transform(X, y)
        actual_X, actual_y = classifier_wrapper._take_apart_X(conflated_X)

        cv = KFold(n_splits=n_splits, random_state=random_seed, shuffle=True)

        # We set min_impurity_decrease depending on the number of instances to obtain a useful feature selection result.
        # min_impurity_decrease was determined based on a series of manual experiments with a varying number of features
        # producing random and zero values. For 1e3 instances, values between 1e-7 and 1e-1 were tested, and 0.0015
        # produced plots closest to the optimal expected result (i.e. significant peak around the number of non-garbage
        # features). Similar experiments were conducted for 1e4 and 1e5 instances. We interpolate between these data points.
        num_instances = len(actual_y)
        xp = np.log10([1e3, 1e5])
        fp = np.log10([0.0015, 0.00025])
        min_impurity_decrease = 10**np.interp(np.log10(num_instances), xp, fp)
        random_forest_clf.set_params(min_impurity_decrease=min_impurity_decrease)

        logger.info("Running feature selection...")
        selector = RFECV(estimator=random_forest_clf,
                         n_jobs=config_global[MAX_CORES],
                         cv=cv,
                         scoring="f1_weighted",  # use f1_weighted because we have very imbalanced data
                         verbose=1)
        selector.fit(actual_X, actual_y)
        logger.info("Done.")

        feature_names = get_feature_names_from_pipeline(pipeline)
        support = selector.support_
        grid_scores = selector.grid_scores_
        assert len(support) == len(feature_names)

        return feature_names, support, grid_scores

    # When using oracle mention pair generation, a randomly determined subset of all mention pairs is used. This has a
    # big influence on the results. We therefore make sure run multiple RFECV iterations with different random seeds for
    # the mention pair generation and aggregate those.
    results = []
    for seed in range(7):
        results.append(run_rfecv_iteration(seed))
    feature_names, supports, grid_scores = list(zip(*results))

    # assert that all results are compatible
    assert len(set(len(s) for s in supports)) == 1
    assert len(set(get_dict_hash(fn) for fn in feature_names)) == 1

    # collect selections in DataFrame
    selections = pd.DataFrame(np.vstack(supports).transpose(), index=pd.Index(feature_names[0], name="feature-name"))
    selected_features = selections.loc[selections.mean(axis=1) > 0.5].index.values

    # write to file(s)
    selections.to_csv(str(serialization_dir / "selected_features_unaggregated.csv"))
    with (serialization_dir / "selected_features.txt").open("w") as f:
        f.write("\n".join(selected_features))
    logger.info("Selected features: " + "\n".join(selected_features))

    # collect scores
    df_grid_scores = []
    for m in grid_scores:
        # number of features and CV-score for that number of features
        x_and_y = np.vstack([np.arange(1, len(m) + 1), m]).transpose()
        df_grid_scores.append(x_and_y)
    df_grid_scores = pd.DataFrame(np.vstack(df_grid_scores))
    df_grid_scores.columns = ["num-features", "weighted-f1"]

    df_grid_scores.to_csv(str(serialization_dir / "grid_scores.csv"))

    # plot feature selection results
    plot_destination = serialization_dir / "rfecv_plot.png"
    ax = sns.lineplot(x="num-features", y="weighted-f1", data=df_grid_scores)
    fig = ax.get_figure()
    fig.savefig(str(plot_destination))