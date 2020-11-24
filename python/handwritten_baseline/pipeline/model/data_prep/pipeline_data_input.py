from logging import Logger
from pathlib import Path
from typing import Optional, Union, List, Tuple, Set, Dict

import numpy as np
import pandas as pd
from numpy.random.mtrand import RandomState
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state

from python import TOPIC_ID, SUBTOPIC, DOCUMENT_ID, EVENT, EVENT_ID
from python.handwritten_baseline.pipeline.data.base import Dataset
from python.handwritten_baseline.pipeline.model.data_prep.mention_pair_generator import MentionPairGenerator
from python.util.util import get_date_based_subdirectory


class MentionPairGeneratorStage(BaseEstimator, TransformerMixin):
    """
    First step of the pipeline which converts a dataset, documents and mentions into a useful set of mention pairs.
    """

    def __init__(self,
                 mpg_training_config: Optional[Dict],
                 mpg_prediction_config: Optional[Dict],
                 random_state: Union[None, int, RandomState],
                 serialization_dir: Optional[Path] = None):
        """

        :param random_state: used when shuffling instances, and for sampling training instances
        """
        self.mpg_training_config = {} if mpg_training_config is None else mpg_training_config
        self.mpg_prediction_config = {} if mpg_prediction_config is None else mpg_prediction_config
        self.random_state = random_state
        self.serialization_dir = serialization_dir

    @staticmethod
    def _check_topics_in_X(X: List[Tuple[Dataset, pd.DataFrame, pd.DataFrame]]):
        all_topic_ids_encountered = set()
        for X_part in X:
            _, documents, _ = X_part
            all_topic_ids_encountered |= set(documents.index.get_level_values(TOPIC_ID).unique().values)
        return sorted(all_topic_ids_encountered)

    def fit_transform(self,
                      X: List[Tuple[Dataset, pd.DataFrame, pd.DataFrame]],
                      y: Optional[List[pd.Series]] = None,
                      **fit_params):
        """
        Called during training, with gold labels
        :param X: partitions of a CDCR dataset from which mention pairs will be generated independently (!)
        :param y: gold events corresponding to the partitions in X
        :param fit_params:
        :return: self
        """
        print("Training with mention pairs from topic(s) " + ", ".join(str(i) for i in self._check_topics_in_X(X)))

        generator = MentionPairGenerator(**self.mpg_training_config,
                                         serialization_dir=get_date_based_subdirectory(self.serialization_dir))

        return convert_X_and_y_to_internal_pipeline_input_fmt(generator=generator,
                                                              X=X,
                                                              y=y,
                                                              random_state=self.random_state)



    def transform(self, X: List[Tuple[Dataset, pd.DataFrame, pd.DataFrame]], y: Optional[List[pd.Series]] = None):
        """
        Called when predicting
        :param X:
        :param y: should be None at prediction time
        :return:
        """
        if y is not None:
            raise ValueError("This method should only be called at prediction time, i.e. without y. With y, fit_transform should have been called.")

        print("Predicting with mention pairs from topic(s) " + ", ".join(str(i) for i in self._check_topics_in_X(X)))

        generator = MentionPairGenerator(**self.mpg_prediction_config,
                                         serialization_dir=get_date_based_subdirectory(self.serialization_dir))

        return convert_X_and_y_to_internal_pipeline_input_fmt(generator=generator,
                                                              X=X,
                                                              y=y,
                                                              random_state=PAIR_PREDICTION_RANDOM_SEED)

NO_LABEL = 0xDEADBEEF

# don't use true randomness for prediction to avoid risks from different states thanks to multiprocessing
PAIR_PREDICTION_RANDOM_SEED = 0

def convert_X_and_y_to_internal_pipeline_input_fmt(generator: MentionPairGenerator,
                                                   X: List[Tuple[Dataset, pd.DataFrame, pd.DataFrame]],
                                                   y: Optional[List[pd.Series]],
                                                   random_state: Union[None, int, RandomState]) -> Tuple[Dataset, np.array, np.array, Set[Tuple[str, int]]]:
    """
    Given our X and y pipeline input data format created by `get_X_and_y_for_pipeline`, generates mention pairs and
    returns the data in the X and y format used *internally* in our pipeline. Oof.
    This is primarily used from `MentionPairGeneratorStage` to generate pairs for training/prediction. We also need this
    method in case we run the pipeline without clustering. Then, `MentionPairScoring` calls this function to obtain the
    gold labels.
    :param generator: generates pairs & labels
    :param X: partitions with input data
    :param y: partitions with labels of input data
    :param random_state:
    :return: a tuple of:
             - Dataset object (for features extractors to pull more information from)
             - mention pairs as a numpy array of dtype str with shape (n, 4) which contains two mention identifiers
             - numpy array of dtype int with shape (n, 1) with the labels corresponding to the mention pairs, this is
               always present (never None) and is filled with NO_LABEL during prediction
             - a set of all unique mention identifiers present in `mention pairs`, so that feature extractors can
               preprocess something for each unique mention
    """

    # assert some assumptions
    assert y is None or len(X) == len(y)
    # The dataset object should be the same in all given partitions, hence take the first object and assert that all
    # later dataset objects are the same one
    dataset = X[0][0]
    for X_part in X:
        assert len(X_part) == 3
        _dataset, _, _ = X_part
        assert dataset == dataset

    # partitions are merged again to allow for extraction of coref links across partitions.
    documents = []
    mentions = []
    for X_part in X:
        _, part_documents, part_mentions = X_part
        documents.append(part_documents)
        mentions.append(part_mentions)
    documents = pd.concat(documents).sort_index()
    mentions = pd.concat(mentions).sort_index()

    mentions_to_gold_events = None
    if y is not None:
        mentions_to_gold_events = pd.concat(y).sort_index()

    # generate pairs - for correct randomness handling, see https://scikit-learn.org/stable/developers/develop.html#random-numbers
    pairs, pair_labels = generator.generate(documents,
                                            mentions,
                                            mentions_to_gold_events=mentions_to_gold_events,
                                            random_state=check_random_state(random_state))

    # convert pairs/labels to np.arrays for easier shuffling/permutation
    all_pairs = np.array(pairs).reshape((-1, 4))
    if pair_labels is not None:
        all_pair_labels = np.array(pair_labels).reshape((-1, 1))
    else:
        # Further down in the pipeline, we use FeatureUnions to join the gold labels of each mention pair with the
        # feature matrix. When we predict, gold labels obviously do not exist, but we can't set the whole array to None
        # here, otherwise the FeatureUnion join would fail. So at prediction time we still need to return a valid array
        # but we fill it with dummy values. This is checked again in `PredictOnTransformClassifierWrapper`.
        all_pair_labels = np.full(shape=(all_pairs.shape[0], 1), fill_value=NO_LABEL)

    # determine unique mention identifiers, so that feature processing can be optimized
    unique_mentions = np.unique(all_pairs.reshape((-1, 2)), axis=0)
    unique_mentions = set((doc_id, int(mention_id_str)) for doc_id, mention_id_str in unique_mentions.tolist())

    # Note that by returning `labels` here, we return information from y in X. (Plus of course the original labels
    # inside `dataset` that we never bothered to remove). The current sklearn pipeline implementation leaves us no
    # other choice, see https://github.com/scikit-learn/scikit-learn/issues/4143
    return dataset, all_pairs, all_pair_labels, unique_mentions


def get_X_and_y_for_pipeline(logger: Logger,
                             dataset: Dataset,
                             doc_partitioning: Union[None, str],
                             oracle_mention_pair_generation: bool) -> Tuple[List[Tuple[Dataset, pd.DataFrame, pd.DataFrame]], List[pd.Series]]:
    """
    This method produces the X and y sequences for the sklearn pipeline. Each element of X consists of a set of
    documents, their mentions (and the full dataset again, for all the preprocessed additional data we need for
    feature extraction). The corresponding y element is a pd.Series which maps mention identifiers to gold events.
    Partitioning of the dataset by documents has two uses: (1) During cross-validation, we can train on a subset of all
    (sub)topics and test on another set of (sub)topics. This achieves greater separation than mixing all possible
    mention pairs. (2) At prediction time, for the sake of speed or analysis, we can apply the mention pair
    classification on gold topics or gold subtopics.
    :param logger:
    :param dataset: dataset instance
    :param doc_partitioning: several options:
                     - if None, return a single (X,y) pair, i.e. no document clustering
                     - if "gold_topics", create one (X,y) pair for each gold topic
                     - if "gold_subtopics", create one (X,y) pair for each gold subtopic
                     - if "gold_auto", partition on gold topics unless there is only one topic, then partition based on
                       gold subtopics
    :param oracle_mention_pair_generation: Enables advanced mention pair generation which uses gold labels even at
                                           prediction time. Do not use for the final evaluation!
    :return: X and y
    """

    if doc_partitioning is None:
        doc_cluster_iterator = [(None, dataset.documents)]
    else:
        num_topics = len(dataset.documents.index.get_level_values(TOPIC_ID).unique())

        do_doc_partitioning_on = doc_partitioning
        if doc_partitioning == "gold_auto":
            do_doc_partitioning_on = "gold_subtopics" if num_topics == 1 else "gold_topics"

        if do_doc_partitioning_on == "gold_topics":
            if num_topics == 1:
                logger.warning("There is only one topic! Only a single (X, y) pair will be created.")
            doc_cluster_iterator = dataset.documents.groupby(TOPIC_ID)
        elif do_doc_partitioning_on == "gold_subtopics":
            doc_cluster_iterator = dataset.documents.groupby([TOPIC_ID, SUBTOPIC])
        else:
            raise ValueError(f"Unknown document partitioning strategy '{do_doc_partitioning_on}'")

    if oracle_mention_pair_generation:
        logger.warning("Using oracle mention pair generation. Pipeline inputs X will contain info from y. Do not use for final evaluation.")

    # for each topic cluster, obtain mentions and the gold coref clustering -> these are our high-level X and y
    X = []
    y = []
    for idx, documents in doc_cluster_iterator:
        mentions = dataset.mentions_action.loc[documents[DOCUMENT_ID]].sort_index()

        mentions_to_gold_events = mentions[EVENT]
        # create blind mentions if oracle is disabled
        if not oracle_mention_pair_generation:
            mentions = mentions.drop(columns=EVENT)

        # we need to pass the full dataset object too
        X.append((dataset, documents, mentions))
        y.append(mentions_to_gold_events)
    return X, y


def create_gold_clustering(mentions_to_gold_events: pd.Series):
    """

    :param mentions_to_gold_events: pandas Series with mention identifiers as the index and events (actual strings, not
                                    the int-based clusters) as the values
    :return: int-based gold clustering as used for evaluation etc.
    """
    event_ids, _ = mentions_to_gold_events.factorize()
    return pd.Series(event_ids, index=mentions_to_gold_events.index, name=EVENT_ID)