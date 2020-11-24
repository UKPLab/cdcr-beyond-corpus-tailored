import pprint
import tempfile
import warnings
from logging import Logger
from pathlib import Path
from typing import Dict, Tuple, List, Callable, Optional

import pandas as pd
from numpy.random.mtrand import RandomState
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.utils import check_random_state

from python.handwritten_baseline.pipeline.model.classifier_clustering.clustering import ScipyClustering
from python.handwritten_baseline.pipeline.model.classifier_clustering.pairwise_classifier_wrapper import \
    PredictOnTransformClassifierWrapper
from python.handwritten_baseline.pipeline.model.classifier_clustering.xgboost import ConvenientXGBClassifier
from python.handwritten_baseline.pipeline.model.data_prep.pipeline_data_input import MentionPairGeneratorStage
from python.handwritten_baseline.pipeline.model.feature_extr import TFIDF_EXTR, LEMMA_EXTR, \
    FEATURE_EXTRACTOR_FEATURE_NAME_SEPARATOR, TIME_EXTR, LOCATION_EXTR, SENTENCE_EMBEDDING_EXTR, \
    ACTION_PHRASE_EMBEDDING_EXTR, WIKIDATA_EMBEDDING_EXTR
from python.handwritten_baseline.pipeline.model.feature_extr.action_phrase import LemmaFeatureExtractor
from python.handwritten_baseline.pipeline.model.feature_extr.embedding_distance.action_phrase_embedding_distance import \
    ActionPhraseEmbeddingDistanceFeature
from python.handwritten_baseline.pipeline.model.feature_extr.embedding_distance.sentence_embedding_distance import \
    SentenceEmbeddingDistanceFeatureExtractorPipelineCreator
from python.handwritten_baseline.pipeline.model.feature_extr.embedding_distance.wikidata_embedding_distance import \
    WikidataEmbeddingDistanceFeatureExtractorPipelineCreator
from python.handwritten_baseline.pipeline.model.feature_extr.tfidf import TfidfFeatureExtractor
from python.handwritten_baseline.pipeline.model.feature_extr.time_and_space.spatial_distance import \
    LocationFeatureExtractorPipelineCreator
from python.handwritten_baseline.pipeline.model.feature_extr.time_and_space.temporal_distance import \
    TimeFeatureExtractorPipelineCreator
from python.handwritten_baseline.pipeline.model.scripts import _TYPE, _ARGS, _KWARGS, _FIT_PARAMS, FEATURE_EXTRACTOR, \
    FEATURE_NAME
from python.handwritten_baseline.pipeline.model.scripts.scoring import CrossDocCorefScoring, MentionPairScoring
from python.util.util import get_dict_hash

# for silencing joblib complaining when it serializes large objects
warnings.simplefilter("ignore", UserWarning)


CLASSIFIER_PIPELINE_STEP_NAME = "classifier"
CLUSTERING_PIPELINE_STEP_NAME = "mention_pair_clustering"

def instantiate_classifier(classifier_config: Dict, random_state: RandomState) -> Tuple:
    classifier_map = {"RandomForest": RandomForestClassifier,
                      "SGDClassifier": SGDClassifier,
                      "ConvenientXGBClassifier": ConvenientXGBClassifier,
                      "MLPClassifier": MLPClassifier}
    classifier_type = classifier_config[_TYPE]
    if not classifier_type in classifier_map:
        raise ValueError("Unknown classifier type: " + classifier_type)
    ClassifierClass = classifier_map[classifier_type]
    classifier = ClassifierClass(*classifier_config.pop(_ARGS, []), random_state=random_state, **classifier_config.pop(
        _KWARGS, {}))
    classifier_fit_params = classifier_config.pop(_FIT_PARAMS, {})
    return classifier, classifier_fit_params


def instantiate_feature_extractors(config: Dict,
                                   use_caching: bool) -> List:
    """

    :param config: dictionary containing configuration for classifier, feature, etc.
    :param use_caching: Whether outputs of transform() calls should be cached. Costly for the first execution, pays off
                        for many calls.
    :return:
    """
    feature_extr_map = {TFIDF_EXTR: TfidfFeatureExtractor,
                        LEMMA_EXTR: LemmaFeatureExtractor,
                        TIME_EXTR: TimeFeatureExtractorPipelineCreator,
                        LOCATION_EXTR: LocationFeatureExtractorPipelineCreator,
                        SENTENCE_EMBEDDING_EXTR: SentenceEmbeddingDistanceFeatureExtractorPipelineCreator,
                        ACTION_PHRASE_EMBEDDING_EXTR: ActionPhraseEmbeddingDistanceFeature,
                        WIKIDATA_EMBEDDING_EXTR: WikidataEmbeddingDistanceFeatureExtractorPipelineCreator}

    feature_extractors_config = config["extractors"]

    # assemble dataframe of selected features, if present in the config
    selected_features = config["selected_features"]
    if selected_features is not None:
        if not selected_features:
            raise ValueError("'selected_features' config parameter is empty. It must be None or a list with at least one item.")

        # Dataframe with features selected during feature selection. Only features mentioned in this dataframe will be
        # returned by feature extractors.
        df = pd.Series(selected_features).str.split(FEATURE_EXTRACTOR_FEATURE_NAME_SEPARATOR, expand=True)
        df.columns = [FEATURE_EXTRACTOR, FEATURE_NAME]
        selected_features = df

    # sanity checks
    for f_extr in feature_extractors_config.keys():
        if f_extr not in feature_extr_map.keys():
            raise ValueError("Unknown feature extractor in config: " + f_extr)
    if selected_features is not None:
        # ensure we don't try to select features from feature extractors we don't have
        for f_extr in selected_features[FEATURE_EXTRACTOR].unique():
            if f_extr not in feature_extr_map.keys():
                raise ValueError("Selected feature extractors contain unknown extractor: " + f_extr)
    feature_extractors = []
    for f_extr_name, feature_extr_config in feature_extractors_config.items():
        FeatureExtractorClass = feature_extr_map[f_extr_name]

        if selected_features is not None:
            features_to_select = selected_features.loc[selected_features[FEATURE_EXTRACTOR] == f_extr_name, FEATURE_NAME].to_list()

            # if no features from this extractor were selected, we can skip creating it entirely
            if not features_to_select:
                continue
            feature_extr_config["features_to_select"] = features_to_select

        if use_caching:
            feature_extr_config["use_cache"] = True

        feature_extractor = FeatureExtractorClass.from_params(feature_extr_config)
        feature_extractors.append((f_extr_name, feature_extractor))
    assert feature_extractors

    return feature_extractors


def get_mention_pair_labels_from_X(X: Tuple):
    dataset, pairs, labels, unique_mentions = X
    return labels


def get_mention_pair_identifiers_from_X(X: Tuple):
    dataset, pairs, labels, unique_mentions = X
    return pairs


def instantiate_pipeline(logger: Logger,
                         config: Dict,
                         with_clustering: bool = False,
                         use_caching: bool = False,
                         scorer_should_return_single_scalar: bool = False,
                         serialization_dir: Optional[Path] = None) -> Tuple[Pipeline, Callable]:
    """
    Uses the entries of a config dictionary to instantiate a scikit-learn pipeline. Additionally returns the scoring
    function to use.
    :param logger:
    :param config: config dictionary
    :param with_clustering: If True, the pipeline will include agglomerative clustering, if False, only mention pair
                            classification is done. The scoring function depends on this choice.
    :param use_caching: Whether fit() calls for all pipeline steps, and transform() calls for features should be cached.
    :param scorer_should_return_single_scalar: If True, the scoring function will return only a single metric as a
                                               scalar. This is useful for running cross-validation. If False, more
                                               metrics are returned as a pd.Series.
    :param serialization_dir: optional serialization dir, only used for debugging
    :return: sklearn pipeline and the scoring function to use for evaluation
    """
    random_seed = config.pop("random_seed")
    random_state = check_random_state(random_seed)

    pairs_config = config.pop("pairs")
    feature_extractors_config = config.pop("features")
    classifier_config = config.pop("classifier")

    # We make use of joblib's caching feature implemented for pipelines in sklearn. joblib only checks if it has seen
    # a pipeline transformer's function arguments before, so we need to make sure to create separate caches when
    # mention pair sampler, feature or classifier config parameters are changed. We use config dict hashes for that.
    if use_caching:
        config_hashes = [get_dict_hash(pairs_config), get_dict_hash(feature_extractors_config)]
        if with_clustering:
            # clustering additionally depends on the classifier config
            config_hashes += [get_dict_hash(classifier_config)]
        pipeline_cache = Path(tempfile.gettempdir()) / ("pipeline_" +  "_".join(config_hashes))
        memory = str(pipeline_cache)
    else:
        memory = None

    # instantiate some bits
    feature_extractors = instantiate_feature_extractors(feature_extractors_config, use_caching)
    classifier, classifier_fit_params = instantiate_classifier(classifier_config, random_state)

    # instantiate mention pair generator stage, which shares parameters with the mention pair scorer (if we use that)
    mpg_training_config = pairs_config.pop("mpg_training")
    mpg_prediction_config = pairs_config.pop("mpg_prediction")
    if pairs_config:
        raise ValueError("Leftover 'pairs' config entries: " + pprint.pformat(pairs_config))

    if with_clustering and mpg_prediction_config is not None:
        # Reasoning: Our mention pair generation parameters only affect the number and distribution of pairs, the
        # number of distribution of mentions is unchanged. Tweaking the mention pair generation process is therefore
        # only useful when the evaluation directly on the pairs, not on the mentions. For clustering, we evaluate based
        # on mentions, and we need distances between all mention pairs, therefore it does not make any sense to use
        # tweaked mention pair sampling at prediction time.
        raise ValueError("'mpg_prediction' cannot be used with clustering!")

    pair_generation_stage = MentionPairGeneratorStage(mpg_training_config,
                                                      mpg_prediction_config,
                                                      random_state=random_state,
                                                      serialization_dir=serialization_dir / "mpg" if serialization_dir is not None else None)
    if with_clustering:
        # using only the most discriminating metric (LEA) is faster when running cross-validation
        scorer = CrossDocCorefScoring(only_lea_f1_for_cv=scorer_should_return_single_scalar)
    else:
        scorer = MentionPairScoring(mpg_prediction_config,
                                    return_neg_log_loss_for_cv=scorer_should_return_single_scalar,
                                    serialization_dir=serialization_dir / "scorer" if serialization_dir is not None else None)

    # Now it's time to assemble the pipeline.
    # For reference, this is the sequence of calls on a pipeline with stages [a, b]:
    #   training:
    #     a: fit called
    #     a: transform
    #     b: fit called
    #   estimating:
    #     a: transform called
    #     b: predict called

    # Combine all feature extractors in a feature union, remove mean and normalize variance.
    feature_extraction_pipeline_steps = [
        ("features", FeatureUnion(feature_extractors)),
        ("scaling", StandardScaler())
    ]

    # This section merges the boolean label of each mention pair with the feature matrix and passes it to the
    # classifier.
    mention_pair_distance_pipeline_steps = [
        ("join_labels_and_feature_matrix", FeatureUnion([
            ("get_labels_from_X", FunctionTransformer(get_mention_pair_labels_from_X)),
            ("feature_extraction", Pipeline(feature_extraction_pipeline_steps)),
        ])),
        (CLASSIFIER_PIPELINE_STEP_NAME, PredictOnTransformClassifierWrapper(classifier, classifier_fit_params)),
    ]

    if with_clustering:
        # create clustering step
        clustering_config = config.pop("clustering")
        clustering = ScipyClustering.from_params(clustering_config)

        # When clustering, we start with generating pairs, then we classify those pairs (see above), but we additionally
        # need to retain the two mention identifiers of each mention pair. Mention pair identifiers and their distance
        # (between 0 and 1) are merged with a FeatureUnion. This "feature matrix" is pulled apart in the clustering step
        # where mentions are clustered agglomeratively according to their pairwise distances.
        pipeline = Pipeline([
            ("pair_generation", pair_generation_stage),
            ("mention_pair_distance_with_identifiers", FeatureUnion([
                ("get_mention_pair_identifiers_from_X", FunctionTransformer(get_mention_pair_identifiers_from_X)),
                ("mention_pair_distance", Pipeline(mention_pair_distance_pipeline_steps))
            ])),
            (CLUSTERING_PIPELINE_STEP_NAME, clustering)
        ], memory=memory)
    else:
        if "clustering" in config:
            logger.warning("Clustering configuration will not be used.")
            config.pop("clustering")

        # In the simplified case, we only need to pass the generated mention pairs to the classification pipeline part.
        pipeline = Pipeline([
            ("pair_generation", pair_generation_stage),
            *mention_pair_distance_pipeline_steps
        ], memory=memory)

    if config:
        raise ValueError("Leftover config entries: " + pprint.pformat(config))

    return pipeline, scorer