from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import traceback

from python import TOPIC_ID, SUBTOPIC, DOCUMENT_ID, MENTION_TYPE, MENTION_TYPES_ACTION, MENTION_TYPES_TIME, \
    MENTION_TYPES_LOCATION, MENTION_TYPES_PARTICIPANTS
from python.pipeline.pipeline import PipelineStage
from python.util.pandas import are_dataframe_indices_compatible


class Dataset:
    def __init__(self,
                 documents: pd.DataFrame,
                 tokens: pd.DataFrame,
                 mentions_action: pd.DataFrame,
                 mentions_time: Optional[pd.DataFrame] = None,
                 mentions_location: Optional[pd.DataFrame] = None,
                 mentions_participants: Optional[pd.DataFrame] = None,
                 mentions_other: Optional[pd.DataFrame] = None,
                 event_vocabs: Dict[str, Any] = None,
                 semantic_roles: Optional[pd.DataFrame] = None):
        """

        :param documents:
        :param tokens:
        :param mentions_action:
        :param mentions_time:
        :param mentions_location:
        :param mentions_participants:
        :param mentions_other:
        :param event_vocabs:
        :param semantic_roles:
        """
        self.documents = documents
        self.tokens = tokens
        self.mentions_action = mentions_action
        self.mentions_time = mentions_time
        self.mentions_location = mentions_location
        self.mentions_participants = mentions_participants
        self.mentions_other = mentions_other
        self.semantic_roles = semantic_roles
        self._event_vocabs = event_vocabs
        self._more = {}

    @property
    def documents(self):
        return self._documents

    @documents.setter
    def documents(self, value: pd.DataFrame):
        # documents must have these index names
        assert value.index.names == [TOPIC_ID, SUBTOPIC, DOCUMENT_ID]

        # all index levels must be of type string
        assert all(ptypes.is_string_dtype(value.index.get_level_values(i).dtype) for i in range(len(value.index.levshape)))

        # index must be sorted, this has caused trouble in the past
        if value is not None:
            assert value.index.is_monotonic_increasing

        self._documents = value

    @property
    def tokens(self):
        return self._tokens

    @tokens.setter
    def tokens(self, value):
        self._tokens = value

    @property
    def mentions_action(self):
        return self._mentions_action

    @staticmethod
    def _assert_all_mention_types_known(df: pd.DataFrame, expected_types: List[str]):
        if df is not None and MENTION_TYPE in df.columns:
            mention_types_used = list(df[MENTION_TYPE].unique())
            if np.nan in mention_types_used:
                print("NaN mention type found. This is fine when encountered during the entity linking stage or later\nOriginating call: " + traceback.format_stack()[-3])
                mention_types_used.remove(np.nan)
            assert all(t in expected_types for t in mention_types_used)

    @mentions_action.setter
    def mentions_action(self, value: pd.DataFrame):
        Dataset._assert_all_mention_types_known(value, MENTION_TYPES_ACTION)

        # index must be sorted, this has caused trouble in the past
        if value is not None:
            assert value.index.is_monotonic_increasing

        self._mentions_action = value

    @property
    def mentions_time(self):
        return self._mentions_time

    @mentions_time.setter
    def mentions_time(self, value):
        Dataset._assert_all_mention_types_known(value, MENTION_TYPES_TIME)

        # index must be sorted, this has caused trouble in the past
        if value is not None:
            assert value.index.is_monotonic_increasing

        self._mentions_time = value

    @property
    def mentions_location(self):
        return self._mentions_location

    @mentions_location.setter
    def mentions_location(self, value):
        Dataset._assert_all_mention_types_known(value, MENTION_TYPES_LOCATION)

        # index must be sorted, this has caused trouble in the past
        if value is not None:
            assert value.index.is_monotonic_increasing

        self._mentions_location = value

    @property
    def mentions_participants(self):
        return self._mentions_participants

    @mentions_participants.setter
    def mentions_participants(self, value):
        Dataset._assert_all_mention_types_known(value, MENTION_TYPES_PARTICIPANTS)

        # index must be sorted, this has caused trouble in the past
        if value is not None:
            assert value.index.is_monotonic_increasing

        self._mentions_participants = value

    @property
    def mentions_other(self):
        return self._mentions_other

    @mentions_other.setter
    def mentions_other(self, value):
        # index must be sorted, this has caused trouble in the past
        if value is not None:
            assert value.index.is_monotonic_increasing

        self._mentions_other = value

    @property
    def event_vocabs(self):
        return self._event_vocabs

    @property
    def semantic_roles(self):
        """
        Returns dataframe which connect action phrases in sentences to time, location and participant mentions in the
        same sentence.
        :return:
        """
        return self._semantic_roles

    @semantic_roles.setter
    def semantic_roles(self, value):
        self._semantic_roles = value

    # some methods for setting arbitrary key-value information
    def set(self, key, value):
        self._more[key] = value

    def get(self, key):
        return self._more[key]

    def has(self, key):
        return key in self._more.keys()

    @classmethod
    def merge(cls, a: "Dataset", b:"Dataset") -> "Dataset":
        """
        This method ignores any extra attributes on the datasets! Only the attributes present in the Dataset constructor
        are merged. Also, we don't merge event vocabs, these were never used anyway.
        :param a:
        :param b:
        :return: merged datasets
        """
        assert are_dataframe_indices_compatible(a.documents, b.documents)
        documents = pd.concat([a.documents, b.documents]).sort_index()

        assert are_dataframe_indices_compatible(a.tokens, b.tokens)
        tokens = pd.concat([a.tokens, b.tokens]).sort_index()

        assert are_dataframe_indices_compatible(a.mentions_action, b.mentions_action)
        mentions_action = pd.concat([a.mentions_action, b.mentions_action]).sort_index()

        def merge_event_component_dfs(attr: str):
            assert hasattr(a, attr) and hasattr(b, attr)

            event_components = []
            if getattr(a, attr) is not None:
                event_components.append(getattr(a, attr))
            if getattr(b, attr) is not None:
                event_components.append(getattr(b, attr))
            if len(event_components) == 0:
                event_components = None
            elif len(event_components) == 1:
                event_components = event_components[0]
            elif len(event_components) == 2:
                assert are_dataframe_indices_compatible(*event_components)
                event_components = pd.concat(event_components).sort_index()
            return event_components

        mentions_time = merge_event_component_dfs("mentions_time")
        mentions_participants = merge_event_component_dfs("mentions_participants")
        mentions_location = merge_event_component_dfs("mentions_location")
        mentions_other = merge_event_component_dfs("mentions_other")

        semantic_roles = []
        if a.semantic_roles is not None:
            semantic_roles.append(a.semantic_roles)
        if b.semantic_roles is not None:
            semantic_roles.append(b.semantic_roles)
        if len(semantic_roles) == 0:
            semantic_roles = None
        elif len(semantic_roles) == 1:
            semantic_roles = semantic_roles[0]
        elif len(semantic_roles) == 2:
            assert are_dataframe_indices_compatible(*semantic_roles, indices_must_be_disjunct=False)
            semantic_roles = pd.concat(semantic_roles, ignore_index=True)

        dataset = Dataset(documents,
                          tokens,
                          mentions_action,
                          mentions_time=mentions_time,
                          mentions_location=mentions_location,
                          mentions_participants=mentions_participants,
                          mentions_other=mentions_other,
                          semantic_roles=semantic_roles)
        return dataset


DATA_SRC_PATH = "data_src_path"
DATASET = "dataset"
MODE_REPLACE = "replace"
MODE_EXTEND = "extend"
MODE_INTERSECT = "intersect"


class BaselineDataLoaderStage(PipelineStage):
    """
    Loads data from disk, producing a Dataset object. When using multiple dataset loader in a pipeline, the datasets
    will be merged.
    """

    def __init__(self, pos, config, config_global, logger):
        super(BaselineDataLoaderStage, self).__init__(pos, config, config_global, logger)

    def _load_dataset(self) -> Dataset:
        raise NotImplementedError

    def run(self, live_objects: Dict[str, Any]):
        dataset = self._load_dataset()

        if DATASET in live_objects:
            dataset = Dataset.merge(live_objects[DATASET], dataset)

        live_objects[DATASET] = dataset


class BaselineDataProcessorStage(PipelineStage):

    def __init__(self, pos, config, config_global, logger):
        super(BaselineDataProcessorStage, self).__init__(pos, config, config_global, logger)
        self.mode = config.get("mode", MODE_REPLACE)
        if self.mode not in [MODE_REPLACE, MODE_EXTEND, MODE_INTERSECT]:
            raise ValueError(f"Unknown mode '{self.mode}. Permitted modes are {','.join([MODE_REPLACE, MODE_EXTEND, MODE_INTERSECT])}.")

    def _process_dataset(self,
                         dataset: Dataset,
                         live_objects: Dict) -> Dataset:
        return dataset

    def run(self, live_objects: Dict[str, Any]):
        if not DATASET in live_objects:
            raise ValueError
        live_objects[DATASET] = self._process_dataset(live_objects[DATASET], live_objects)