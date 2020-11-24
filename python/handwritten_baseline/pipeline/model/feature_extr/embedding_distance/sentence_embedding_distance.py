import pprint
from typing import Optional, List, Tuple, Set, Dict

import numpy as np
from overrides import overrides
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from python import SENTENCE_IDX
from python.handwritten_baseline import SENTENCE_EMBEDDINGS
from python.handwritten_baseline.pipeline.data.base import Dataset
from python.handwritten_baseline.pipeline.model.feature_extr import SENTENCE_EMBEDDING_EXTR
from python.handwritten_baseline.pipeline.model.feature_extr.base_mixin import FeatureExtractorMixin
from python.handwritten_baseline.pipeline.model.feature_extr.embedding_distance import create_feature_names
from python.handwritten_baseline.pipeline.model.feature_extr.util import batch_cosine_similarity


class SentenceEmbeddingDistanceFeatureExtractorPipelineCreator:

    @classmethod
    def from_params(cls, config: Dict):
        extractor = SentenceEmbeddingDistanceFeature.from_params(config)
        imputer = SimpleImputer(missing_values=np.nan, strategy="median")
        return make_pipeline(extractor, imputer)


SURROUNDING_SENTENCE = "surrounding-sentence"
DOC_START = "doc-start"


class SentenceEmbeddingDistanceFeature(FeatureExtractorMixin):

    def __init__(self,
                 use_cache: bool,
                 features_to_select: Optional[List[str]]):
        super(SentenceEmbeddingDistanceFeature, self).__init__(SENTENCE_EMBEDDING_EXTR, use_cache, features_to_select)

    @overrides
    def _transform(self, dataset: Dataset, pairs: List[Tuple[Tuple, Tuple]], unique_mentions: Set[Tuple]):
        # obtain embeddings
        assert dataset.has(SENTENCE_EMBEDDINGS)
        sentence_embeddings = dataset.get(SENTENCE_EMBEDDINGS)  # type: Tuple[Dict[Tuple[str, int], int], np.array]
        embedding_index, embedding_mat = sentence_embeddings

        mentions_action = dataset.mentions_action

        # compute a mean embedding in case we need to pad somewhere
        mean_embedding = embedding_mat.mean(axis=0)

        # precompute embedding matrices for each action mention
        precomputed_sentence = {}
        precomputed_doc_start = {}
        for mention_idx in unique_mentions:
            assert len(mention_idx) == 2
            doc_id, mention_id = mention_idx

            # look up sentence embedding of the sentence containing the action mention
            sent_idx_of_action = mentions_action.loc[mention_idx, SENTENCE_IDX]
            surrounding_sent_embedding = embedding_mat[embedding_index[(doc_id, sent_idx_of_action)]]

            # for the document start, take n sentences from the start of the document and concatenate their embeddings
            NUM_SENTENCES_DOC_START = 3
            doc_start_sent_embeddings = []
            for i in range(NUM_SENTENCES_DOC_START):
                # there might be documents shorter than NUM_SENTENCES_DOC_START, therefore check: if there are not
                # enough sentences, pad with the mean embedding
                if (doc_id, i) in embedding_index:
                    sent_embedding = embedding_mat[embedding_index[(doc_id, i)]]
                else:
                    sent_embedding = mean_embedding
                doc_start_sent_embeddings.append(sent_embedding)
            doc_start_embedding = np.hstack(doc_start_sent_embeddings)

            precomputed_sentence[mention_idx] = surrounding_sent_embedding
            precomputed_doc_start[mention_idx] = doc_start_embedding

        feature_columns = []
        for vectors, feature_desc in [(precomputed_sentence, SURROUNDING_SENTENCE), (precomputed_doc_start, DOC_START)]:
            feature_column = batch_cosine_similarity(pairs, vectors, desc=f"{self.name} {feature_desc}")
            feature_columns.append(feature_column)
        feature_matrix = np.hstack(feature_columns)
        return feature_matrix

    @overrides
    def _get_plain_names_of_all_features(self) -> List[str]:
        return create_feature_names([SURROUNDING_SENTENCE, DOC_START], [])

    @classmethod
    @overrides
    def from_params(cls, config: Dict):
        use_cache = config.pop("use_cache", False)
        features_to_select = config.pop("features_to_select", None)
        obj = SentenceEmbeddingDistanceFeature(use_cache, features_to_select)
        if config:
            raise ValueError("Leftover configuration: " + pprint.pformat(config))
        return obj