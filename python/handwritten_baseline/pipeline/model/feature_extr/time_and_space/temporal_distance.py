
import datetime
import pprint
from typing import Optional, List, Tuple, Set, Dict

import numpy as np
import pandas as pd
from overrides import overrides
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from python import DOCUMENT_ID, PUBLISH_DATE, SUBTOPIC, TOPIC_ID, MENTION_TYPE, TIME_OF_THE_DAY, TIME_DATE
from python.handwritten_baseline import TIMEX_NORMALIZED_PARSED, TIME, TIMEX_NORMALIZED
from python.handwritten_baseline.pipeline.data.base import Dataset
from python.handwritten_baseline.pipeline.model.feature_extr import TIME_EXTR, fix_all_nan_columns
from python.handwritten_baseline.pipeline.model.feature_extr.base_mixin import FeatureExtractorMixin
from python.handwritten_baseline.pipeline.model.feature_extr.time_and_space import \
    look_up_document_level_event_component, look_up_event_component_by_sentence, \
    look_up_event_component_by_srl, look_up_event_component_from_closest_preceding_sentence


class TimeFeatureExtractorPipelineCreator:
    """
    The temporal distance feature consists of two pipeline stages: extracting features in the transform() method of
    `TemporalDistanceFeature` followed by imputation of missing values.
    """

    @classmethod
    def from_params(cls, config: Dict):
        extractor = TemporalDistanceFeature.from_params(config)

        fix_nan_columns = FunctionTransformer(fix_all_nan_columns)

        # Our feature extraction returns NaNs in case one of two mentions in a pair has no temporal information, so we
        # need to fill those NaNs. 0 and -1 would be misleading for the classifier, therefore use the median feature
        # value.
        imputer = SimpleImputer(missing_values=np.nan, strategy="median")
        return make_pipeline(extractor, fix_nan_columns, imputer)


class TemporalDistanceFeature(FeatureExtractorMixin):
    """
    Computes temporal distance (hour, day, ...) between temporal expressions of a mention pair. Two variants for finding
    temporal expressions exist: (1) document-level, where we pick the first temporal expression in a document and (2)
    mention-level, where we use SRL to find the temporal expression attached to the mention action, or fall back to the
    first temporal expression in the same sentence or fall back to the closest temporal expression from a previous sentence.
    """

    def __init__(self,
                 use_cache: bool,
                 features_to_select: Optional[List[str]]):
        super(TemporalDistanceFeature, self).__init__(TIME_EXTR, use_cache, features_to_select)


    @staticmethod
    def compute_temporal_distance_features(a_date: Optional[datetime.datetime], b_date: Optional[datetime.datetime]):
        """
        Compute temporal distance between two datetimes per day, month, etc.
        :param a_date:
        :param b_date:
        :return: list of distances
        """
        if a_date is None or b_date is None or pd.isna(a_date) or pd.isna(b_date):
            return [None] * 5
        else:
            # difference in year, month, week, day, hour
            return [abs(i) for i in
                    [a_date.year - b_date.year,
                     a_date.month - b_date.month,
                     a_date.isocalendar()[1] - b_date.isocalendar()[1],
                     a_date.day - b_date.day,
                     a_date.hour - b_date.hour]
                    ]

    @overrides
    def _transform(self, dataset: Dataset, pairs: List[Tuple[Tuple, Tuple]], unique_mentions: Set[Tuple]):
        assert dataset.semantic_roles is not None and dataset.mentions_time is not None

        sr = dataset.semantic_roles
        mentions_time = dataset.mentions_time
        mentions_action = dataset.mentions_action
        documents = dataset.documents

        if PUBLISH_DATE in documents.columns:
            publish_date_by_document = documents[PUBLISH_DATE].map(lambda v: pd.to_datetime(v, errors="coerce"))
            publish_date_by_document.index = publish_date_by_document.index.droplevel(level=[TOPIC_ID, SUBTOPIC])
        else:
            publish_date_by_document = pd.Series(None, index=documents[DOCUMENT_ID], dtype=object)

        # keep only useful TIMEX annotations: dates and times (drop durations, ranges, ...) and actually parsed expressions
        mentions_time = mentions_time.loc[
            mentions_time[MENTION_TYPE].isin([TIME_OF_THE_DAY, TIME_DATE]) & mentions_time[TIMEX_NORMALIZED_PARSED].notna()]

        # precompute relevant information per mention
        precomputed = {}
        for action_mention in unique_mentions:
            assert len(action_mention) == 2  # (doc_id, mention_id)

            # look up temporal information from various regions
            x = (mentions_time, TIME, TIMEX_NORMALIZED)
            temporal_closest_preceding_sentence_level = look_up_event_component_from_closest_preceding_sentence(x, action_mention, mentions_action)
            temporal_sentence_level = look_up_event_component_by_sentence(x, action_mention, mentions_action)
            temporal_srl_level = look_up_event_component_by_srl(x, action_mention, sr)

            # keep track of the time closest to the mention overall too, use that as a feature
            temporal_closest_overall = None

            if temporal_closest_preceding_sentence_level is not None:
                temporal_closest_preceding_sentence_level = temporal_closest_preceding_sentence_level[TIMEX_NORMALIZED_PARSED]
                temporal_closest_overall = temporal_closest_preceding_sentence_level
            if temporal_sentence_level is not None:
                temporal_sentence_level = temporal_sentence_level[TIMEX_NORMALIZED_PARSED]
                temporal_closest_overall = temporal_sentence_level
            if temporal_srl_level is not None:
                temporal_srl_level = temporal_srl_level[TIMEX_NORMALIZED_PARSED]
                temporal_closest_overall = temporal_srl_level

            # start search the other way around: the first mentioned temporal expression in the document
            temporal_doc_level = look_up_document_level_event_component(mentions_time, action_mention)
            if temporal_doc_level is not None:
                temporal_doc_level = temporal_doc_level[TIMEX_NORMALIZED_PARSED]

            # extract document publish date too
            doc_id, _ = action_mention
            publish_datetime = publish_date_by_document.loc[doc_id]

            precomputed[action_mention] = {"srl": temporal_srl_level,
                                           "sentence": temporal_sentence_level,
                                           "closest-preceding-sentence": temporal_closest_preceding_sentence_level,
                                           "closest-overall": temporal_closest_overall,
                                           "doc": temporal_doc_level,
                                           "doc-publish": publish_datetime}

        # compute distances between mention pairs
        list_of_instance_features = []
        for a_idx, b_idx in pairs:
            instance_features = []

            for key in ["srl", "sentence", "closest-preceding-sentence", "closest-overall", "doc", "doc-publish"]:
                instance_features += self.compute_temporal_distance_features(precomputed[a_idx][key],
                                                                             precomputed[b_idx][key])

            list_of_instance_features.append(instance_features)
        feature_matrix = np.vstack(list_of_instance_features).astype(np.float)
        return feature_matrix

    @overrides
    def _get_plain_names_of_all_features(self) -> List[str]:
        names = []
        for level in ["srl", "sentence", "closest-preceding-sentence", "closest-overall", "document", "document-publish"]:
            for metric in ["year", "month", "week", "day", "hour"]:
                names.append("-".join(["distance", level, "level", metric]))
        return names

    @classmethod
    @overrides
    def from_params(cls, config: Dict):
        use_cache = config.pop("use_cache", False)
        features_to_select = config.pop("features_to_select", None)
        obj = TemporalDistanceFeature(use_cache, features_to_select)
        if config:
            raise ValueError("Leftover configuration: " + pprint.pformat(config))
        return obj