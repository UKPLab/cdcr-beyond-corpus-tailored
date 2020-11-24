import pprint
from typing import Optional, List, Tuple, Set, Dict

import numpy as np
import pandas as pd
from overrides import overrides
from scipy.spatial.distance import cosine
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from python import SENTENCE_IDX, DOCUMENT_ID, MENTION_ID
from python.handwritten_baseline import WIKIDATA_QID, WIKIDATA_EMBEDDINGS, ACTION, TIME, LOCATION, \
    PARTICIPANTS, OTHER, MENTION_TYPE_COARSE, MENTION_TEXT, COMPONENT_MENTION_ID
from python.handwritten_baseline.pipeline.data.base import Dataset
from python.handwritten_baseline.pipeline.model.feature_extr import WIKIDATA_EMBEDDING_EXTR, fix_all_nan_columns
from python.handwritten_baseline.pipeline.model.feature_extr.base_mixin import FeatureExtractorMixin
from python.handwritten_baseline.pipeline.model.feature_extr.embedding_distance import create_feature_names, \
    compute_pairwise_embedding_distance_features


class WikidataEmbeddingDistanceFeatureExtractorPipelineCreator:

    @classmethod
    def from_params(cls, config: Dict):
        extractor = WikidataEmbeddingDistanceFeature.from_params(config)

        fix_nan_columns = FunctionTransformer(fix_all_nan_columns)
        imputer = SimpleImputer(missing_values=np.nan, strategy="median")
        return make_pipeline(extractor, fix_nan_columns, imputer)


ACTION_MENTION = "action-mention"
SEMANTIC_ROLE_ARGS = "semantic-role-args"
SURROUNDING_SENTENCE = "surrounding-sentence"
SENTENCE_CONTEXT = "sentence-context"
DOC_START = "doc-start"
FEATURES_IN_ORDER = [SEMANTIC_ROLE_ARGS, SURROUNDING_SENTENCE, SENTENCE_CONTEXT, DOC_START]


class WikidataEmbeddingDistanceFeature(FeatureExtractorMixin):

    def __init__(self,
                 use_cache: bool,
                 features_to_select: Optional[List[str]]):
        super(WikidataEmbeddingDistanceFeature, self).__init__(WIKIDATA_EMBEDDING_EXTR, use_cache, features_to_select)

    @overrides
    def _transform(self, dataset: Dataset, pairs: List[Tuple[Tuple, Tuple]], unique_mentions: Set[Tuple]):
        # obtain embeddings
        assert dataset.has(WIKIDATA_EMBEDDINGS)
        wikidata_embeddings = dataset.get(WIKIDATA_EMBEDDINGS)  # type: Tuple[Dict[str, int], np.array]
        embedding_index, embedding_mat = wikidata_embeddings

        # create one large dataframe of all named entities which are entity linked to Wikidata
        linked_event_components = []
        for mention_type_coarse, df in {ACTION: dataset.mentions_action,
                                        PARTICIPANTS: dataset.mentions_participants,
                                        TIME: dataset.mentions_time,
                                        LOCATION: dataset.mentions_location,
                                        OTHER: dataset.mentions_other}.items():
            if df is None:
                continue

            # keep only entities/mentions which are linked to Wikidata
            linked_subset = df.loc[df[WIKIDATA_QID].notna()]
            # drop those linked embeddings for which we don't have an embedding
            with_embedding = linked_subset.loc[linked_subset[WIKIDATA_QID].isin(embedding_index.keys())]
            # keep only relevant columns
            only_relevant_columns = with_embedding.reindex(columns=[MENTION_TEXT, SENTENCE_IDX, WIKIDATA_QID])

            only_relevant_columns[MENTION_TYPE_COARSE] = mention_type_coarse

            linked_event_components.append(only_relevant_columns)
        linked_event_components = pd.concat(linked_event_components).set_index(MENTION_TYPE_COARSE, append=True)
        assert linked_event_components.index.is_unique

        # convert QID into index of the corresponding embedding in `embedding_mat`
        linked_event_components[WIKIDATA_QID] = linked_event_components[WIKIDATA_QID].map(embedding_index)
        assert linked_event_components[WIKIDATA_QID].notna().all() and not linked_event_components[WIKIDATA_QID].astype(
            str).str.startswith("Q").any()

        linked_event_components = linked_event_components.reset_index()
        mentions_action = dataset.mentions_action
        sr = dataset.semantic_roles

        # precompute embedding matrices for each action mention
        precomputed = {}
        for mention_idx in unique_mentions:
            assert len(mention_idx) == 2
            doc_id, mention_id = mention_idx

            linked_in_doc = linked_event_components.loc[linked_event_components[DOCUMENT_ID] == doc_id]

            # look up embedding for action mention (rarely the case)
            linked_action_mention = linked_in_doc.loc[
                (linked_in_doc[MENTION_TYPE_COARSE] == ACTION) & (linked_in_doc[MENTION_ID] == mention_id)]
            if not linked_action_mention.empty:
                action_mention_embedding = embedding_mat[linked_action_mention[WIKIDATA_QID].values]
            else:
                action_mention_embedding = None

            # if available, create matrix of embeddings from all entity linked SRL arguments
            srl_args_of_mention = sr.loc[(sr[DOCUMENT_ID] == doc_id) & (sr[MENTION_ID] == mention_id)]
            if not srl_args_of_mention.empty:
                linked_srl_args_for_mention = srl_args_of_mention.merge(linked_in_doc,
                                                                        left_on=[COMPONENT_MENTION_ID,
                                                                                 MENTION_TYPE_COARSE],
                                                                        right_on=[MENTION_ID,
                                                                                  MENTION_TYPE_COARSE]).drop_duplicates(WIKIDATA_QID)
                linked_srl_embeddings = embedding_mat[linked_srl_args_for_mention[WIKIDATA_QID].values]
            else:
                linked_srl_embeddings = None

            # create matrix of embeddings from all linked entities in the same sentence as the action mention
            sent_idx_of_action = mentions_action.loc[mention_idx, SENTENCE_IDX]

            linked_in_surrounding_sent = linked_in_doc.loc[
                linked_in_doc[SENTENCE_IDX] == sent_idx_of_action].drop_duplicates(WIKIDATA_QID)
            if not linked_in_surrounding_sent.empty:
                surrounding_sent_embeddings = embedding_mat[linked_in_surrounding_sent[WIKIDATA_QID].values]
            else:
                surrounding_sent_embeddings = None

            # create matrix of embeddings from all linked entities in the context of the action mention
            NUM_SENTENCES_CONTEXT = 2
            sent_idx_from = sent_idx_of_action - NUM_SENTENCES_CONTEXT
            sent_idx_to = sent_idx_of_action + NUM_SENTENCES_CONTEXT

            linked_in_context = linked_in_doc.loc[(linked_in_doc[SENTENCE_IDX] >= sent_idx_from) & (
                    linked_in_doc[SENTENCE_IDX] <= sent_idx_to)].drop_duplicates(WIKIDATA_QID)
            if not linked_in_context.empty:
                context_embeddings = embedding_mat[linked_in_context[WIKIDATA_QID].values]
            else:
                context_embeddings = None

            # create matrix of embeddings from linked entities at the document start
            NUM_SENTENCES_DOC_START = 3
            linked_at_doc_start = linked_in_doc.loc[
                (linked_in_doc[SENTENCE_IDX] < NUM_SENTENCES_DOC_START)].drop_duplicates(WIKIDATA_QID)
            if not linked_at_doc_start.empty:
                doc_start_embeddings = embedding_mat[linked_at_doc_start[WIKIDATA_QID].values]
            else:
                doc_start_embeddings = None

            precomputed[mention_idx] = {ACTION_MENTION: action_mention_embedding,
                                        SEMANTIC_ROLE_ARGS: linked_srl_embeddings,
                                        SURROUNDING_SENTENCE: surrounding_sent_embeddings,
                                        SENTENCE_CONTEXT: context_embeddings,
                                        DOC_START: doc_start_embeddings}

        # using the precomputed action mention representations, compute pairwise features
        list_of_instance_features = []
        for pair in pairs:
            a_idx, b_idx = pair

            instance_features = []

            # compute distance between action mention embeddings
            a_action_mention_mat = precomputed[a_idx][ACTION_MENTION]
            b_action_mention_mat = precomputed[b_idx][ACTION_MENTION]
            if a_action_mention_mat is None or b_action_mention_mat is None:
                instance_features.append(None)
            else:
                instance_features.append(cosine(a_action_mention_mat, b_action_mention_mat))

            # the order is important here, it has to match the names in __init__!
            for key in FEATURES_IN_ORDER:
                a_mat = precomputed[a_idx][key]
                b_mat = precomputed[b_idx][key]
                features_of_key = compute_pairwise_embedding_distance_features(a_mat, b_mat)
                instance_features += features_of_key

            instance_features = np.array(instance_features, dtype=self.dtype)
            list_of_instance_features.append(instance_features)

        feature_matrix = np.vstack(list_of_instance_features)
        return feature_matrix

    @overrides
    def _get_plain_names_of_all_features(self) -> List[str]:
        return create_feature_names([ACTION_MENTION], FEATURES_IN_ORDER)

    @classmethod
    @overrides
    def from_params(cls, config: Dict):
        use_cache = config.pop("use_cache", False)
        features_to_select = config.pop("features_to_select", None)
        obj = WikidataEmbeddingDistanceFeature(use_cache, features_to_select)
        if config:
            raise ValueError("Leftover configuration: " + pprint.pformat(config))
        return obj