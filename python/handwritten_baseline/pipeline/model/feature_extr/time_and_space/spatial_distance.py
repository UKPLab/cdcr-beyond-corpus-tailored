import pprint
from typing import Optional, List, Tuple, Set, Dict

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from overrides import overrides
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from python.handwritten_baseline import LATITUDE, LONGITUDE, MENTION_TEXT, LOCATION, GEO_HIERARCHY
from python.handwritten_baseline.pipeline.data.base import Dataset
from python.handwritten_baseline.pipeline.model.feature_extr import LOCATION_EXTR, fix_all_nan_columns
from python.handwritten_baseline.pipeline.model.feature_extr.base_mixin import FeatureExtractorMixin
from python.handwritten_baseline.pipeline.model.feature_extr.time_and_space import \
    look_up_document_level_event_component, look_up_event_component_by_sentence, \
    look_up_event_component_by_srl, look_up_event_component_from_closest_preceding_sentence


class LocationFeatureExtractorPipelineCreator:

    @classmethod
    def from_params(cls, config: Dict):
        extractor = SpatialDistanceFeature.from_params(config)

        fix_nan_columns = FunctionTransformer(fix_all_nan_columns)

        # Our feature extraction returns NaNs in case one of two mentions in a pair has no spatial information, so we
        # need to fill those NaNs. 0 and -1 would be misleading for the classifier, therefore use the median feature
        # value.
        imputer = SimpleImputer(missing_values=np.nan, strategy="median")

        return make_pipeline(extractor, fix_nan_columns, imputer)


class SpatialDistanceFeature(FeatureExtractorMixin):

    def __init__(self,
                 use_cache: bool,
                 features_to_select: Optional[List[str]]):
        super(SpatialDistanceFeature, self).__init__(LOCATION_EXTR, use_cache, features_to_select)

    @staticmethod
    def compute_spatial_distance_features(a_location: Optional[pd.Series], b_location: Optional[pd.Series]):
        """
        Compute spatial distance between two locations
        :param a_location:
        :param b_location:
        :return: list of distances
        """
        if a_location is None or b_location is None:
            return [1.0, None]
        else:
            features = []

            # geo hierarchy feature
            geo_hierarchy_feature = 1.0
            a_hier = a_location[GEO_HIERARCHY]
            b_hier = b_location[GEO_HIERARCHY]
            if type(a_hier) is list and type(b_hier) is list:
                # Determine matching quality between hierarchies: hierarchies are ordered from more specific to less
                # specific. The more specific of a match we can find between the two lists of locations, the higher the
                # chances of coreference. Enumerate each item, match items in both lists, find "earliest" match,
                # normalize enumeration and sum up the two values. Done! Worst case: no match, we return 1.0. Slightly
                # better case: the last entry in each list matches, which gives us a score close to 1. Best case: first
                # entries match, which gives a score of 0.
                matches = set(a_hier) & set(b_hier)
                if matches:
                    scored_matches = [a_hier.index(m) / len(a_hier) + b_hier.index(m) / len(b_hier) for m in matches]
                    geo_hierarchy_feature = min(scored_matches) / 2
            features.append(geo_hierarchy_feature)

            # if exact coordinates are known, compute geodesic distance between coordinates
            geodesic_distance = None
            if pd.notna(a_location[LATITUDE]) and pd.notna(b_location[LATITUDE]):
                a_latlong = tuple(a_location[[LATITUDE, LONGITUDE]].values)
                b_latlong = tuple(b_location[[LATITUDE, LONGITUDE]].values)
                geodesic_distance = geodesic(a_latlong, b_latlong).kilometers
            features.append(geodesic_distance)
            return features

    @overrides
    def _transform(self, dataset: Dataset, pairs: List[Tuple[Tuple, Tuple]], unique_mentions: Set[Tuple]):
        assert dataset.semantic_roles is not None and dataset.mentions_location is not None

        sr = dataset.semantic_roles
        mentions_location = dataset.mentions_location
        mentions_action = dataset.mentions_action

        # precompute relevant information per mention
        precomputed = {}
        for action_mention in unique_mentions:
            assert len(action_mention) == 2  # (doc_id, mention_id)

            # look up spatial information from various regions
            x = (mentions_location, LOCATION, MENTION_TEXT)
            spatial_closest_preceding_sentence_level = look_up_event_component_from_closest_preceding_sentence(x, action_mention, mentions_action)
            spatial_sentence_level = look_up_event_component_by_sentence(x, action_mention, mentions_action)
            spatial_srl_level = look_up_event_component_by_srl(x, action_mention, sr)

            # keep track of the location closest to the mention overall too, use that as a feature
            spatial_closest_overall = None
            if spatial_closest_preceding_sentence_level is not None:
                spatial_closest_overall = spatial_closest_preceding_sentence_level
            if spatial_sentence_level is not None:
                spatial_closest_overall = spatial_sentence_level
            if spatial_srl_level is not None:
                spatial_closest_overall = spatial_srl_level

            # start search the other way around: the first mentioned spatial expression in the document
            spatial_doc_level = look_up_document_level_event_component(mentions_location, action_mention)

            precomputed[action_mention] = {"srl": spatial_srl_level,
                                           "sentence": spatial_sentence_level,
                                           "closest-preceding-sentence": spatial_closest_preceding_sentence_level,
                                           "closest-overall": spatial_closest_overall,
                                           "doc": spatial_doc_level}

        # compute distances between mention pairs
        list_of_instance_features = []
        for a_idx, b_idx in pairs:
            instance_features = []

            for key in ["srl", "sentence", "closest-preceding-sentence", "closest-overall", "doc"]:
                instance_features += self.compute_spatial_distance_features(precomputed[a_idx][key],
                                                                            precomputed[b_idx][key])

            list_of_instance_features.append(instance_features)
        feature_matrix = np.vstack(list_of_instance_features).astype(np.float)
        return feature_matrix

    @overrides
    def _get_plain_names_of_all_features(self) -> List[str]:
        names = []
        for level in ["srl", "sentence", "closest-preceding-sentence", "closest-overall", "document"]:
            for metric in ["geo-hierarchy-match", "geodesic-distance"]:
                names.append("-".join([level, "level", metric]))
        return names

    @classmethod
    @overrides
    def from_params(cls, config: Dict):
        use_cache = config.pop("use_cache", False)
        features_to_select = config.pop("features_to_select", None)
        obj = SpatialDistanceFeature(use_cache, features_to_select)
        if config:
            raise ValueError("Leftover configuration: " + pprint.pformat(config))
        return obj