import pprint
from typing import Optional, List, Tuple, Set, Any, Dict

import numpy as np
from overrides import overrides
from sklearn.feature_extraction.text import TfidfVectorizer

from python import TOKEN, DOCUMENT_ID, SENTENCE_IDX
from python.handwritten_baseline import LEMMA
from python.handwritten_baseline.pipeline.data.base import Dataset
from python.handwritten_baseline.pipeline.model.feature_extr import TFIDF_EXTR
from python.handwritten_baseline.pipeline.model.feature_extr.base_mixin import FeatureExtractorMixin
from python.handwritten_baseline.pipeline.model.feature_extr.util import batch_cosine_similarity


class TfidfFeatureExtractor(FeatureExtractorMixin):
    """
    Computes the TF-IDF similarity between a mention pair. Three variants: (1) TF-IDF between sentence containing the
    mention, (2) TF-IDF between the extended sentence context of a mention and (3) TF-IDF between the full documents the
    mentions are coming from.
    """

    def __init__(self,
                 lowercase: bool,
                 use_lemmas: bool,
                 num_sentence_context: int,
                 use_cache: bool,
                 features_to_select: Optional[List[str]]):
        """

        :param lowercase: apply lowercasing yes or no
        :param use_lemmas: use lemmas or surface forms
        :param num_sentence_context: number of sentences left and right which define the sentence context -> results in
                                     a window of 2*self._num_sentence_context + 1 sentences
        """
        super(TfidfFeatureExtractor, self).__init__(TFIDF_EXTR, use_cache, features_to_select)

        self.lowercase = lowercase
        self.use_lemmas = use_lemmas
        self.num_sentence_context = num_sentence_context

    @staticmethod
    def get_tfidf_of_doc(doc_id: Any, dataset: Dataset, vectorizer_: TfidfVectorizer) -> np.array:
        tokens = dataset.tokens.loc[doc_id, TOKEN].values
        detokenized = " ".join(tokens)
        return vectorizer_.transform([detokenized]).toarray()

    @staticmethod
    def get_tfidf_of_mention_sentence(idx: Tuple, dataset: Dataset, vectorizer_: TfidfVectorizer) -> np.array:
        doc_id, _ = idx
        sent_idx = dataset.mentions_action.at[idx, SENTENCE_IDX]
        tokens = dataset.tokens.loc[(doc_id, sent_idx), TOKEN].values
        detokenized = " ".join(tokens)
        return vectorizer_.transform([detokenized]).toarray()

    @staticmethod
    def get_tfidf_of_mention_context(idx: Tuple, dataset: Dataset, vectorizer_: TfidfVectorizer, num_sentence_context: int) -> np.array:
        doc_id, _ = idx
        sent_idx = dataset.mentions_action.at[idx, SENTENCE_IDX]

        # determine how many preceding and following sentences there are for the mention sentence in this document
        document = dataset.tokens.loc[doc_id, TOKEN]
        sent_idx_start = max(sent_idx - num_sentence_context, 0)
        sent_idx_end = min(sent_idx + num_sentence_context,
                           document.index.get_level_values(SENTENCE_IDX).max())

        tokens = document.loc[slice(sent_idx_start, sent_idx_end)].values
        detokenized = " ".join(tokens)
        return vectorizer_.transform([detokenized]).toarray()

    @overrides
    def _transform(self, dataset: Dataset, pairs: List[Tuple[Tuple, Tuple]], unique_mentions: Set[Tuple]):
        # TFIDF vectorization is an unsupervised transformation, therefore apply it in transform(), not in fit(). It
        # would not make much sense anyway to use a TF-IDF vectorizer trained on train and apply it on test.
        # The recommended way to handle pretokenized text according to the docs is to join with spaces and use
        # whitespace tokenization, see https://scikit-learn.org/stable/modules/feature_extraction.html#customizing-the-vectorizer-classes
        vectorizer_ = TfidfVectorizer(tokenizer=str.split, lowercase=self.lowercase, token_pattern=None, min_df=3, stop_words="english")

        tokens_df = dataset.tokens
        tokens = tokens_df[LEMMA] if self.use_lemmas else tokens_df[TOKEN]

        docs = []
        for doc_id, df in tokens.groupby(DOCUMENT_ID):
            tokens = df.values.tolist()
            docs.append(" ".join(tokens))
        vectorizer_.fit(docs)

        # precompute relevant information per document and mention
        unique_documents = {doc_id for doc_id, _ in unique_mentions}
        precomp_documents = {doc_id: self.get_tfidf_of_doc(doc_id, dataset, vectorizer_) for doc_id in unique_documents}

        precomp_surrounding_sentence = {}
        precomp_context = {}
        for mention_idx in unique_mentions:
            assert len(mention_idx) == 2 # (doc_id, mention_id)

            # features for the mention sentence: check if mentions were detected for both sentences
            surrounding_sentence = self.get_tfidf_of_mention_sentence(mention_idx, dataset, vectorizer_)
            context = self.get_tfidf_of_mention_context(mention_idx, dataset, vectorizer_, self.num_sentence_context)

            precomp_surrounding_sentence[mention_idx] = surrounding_sentence
            precomp_context[mention_idx] = context

        # compute cosine similarity between each pair of vectors to obtain features
        feature_columns = []
        for vectors, feature_desc in [(precomp_documents, "document"),
                                      (precomp_surrounding_sentence, "sentence"),
                                      (precomp_context, "context")]:
            if feature_desc == "document":
                pairs_transform = lambda tup: tup[0]    # our document vectors map from doc-id to np.array
            else:
                pairs_transform = None

            feature_column = batch_cosine_similarity(pairs, vectors, pairs_transform=pairs_transform, desc=f"{self.name} {feature_desc}")
            feature_columns.append(feature_column)
        feature_matrix = np.hstack(feature_columns)
        return feature_matrix

    @overrides
    def _get_plain_names_of_all_features(self) -> List[str]:
        return ["document-similarity", "surrounding-sentence-similarity", "context-similarity"]

    @classmethod
    @overrides
    def from_params(cls, config: Dict):
        # Tested all four combinations in a small CV-experiment, this combination performed best by a small margin.
        lowercase = config.pop("lowercase", True)
        use_lemmas = config.pop("use_lemmas", False)

        num_sentence_context = config.pop("num_sentence_context", 2)

        use_cache = config.pop("use_cache", False)
        features_to_select = config.pop("features_to_select", None)
        obj = TfidfFeatureExtractor(lowercase=lowercase,
                                    use_lemmas=use_lemmas,
                                    num_sentence_context=num_sentence_context,
                                    use_cache=use_cache,
                                    features_to_select=features_to_select)
        if config:
            raise ValueError("Leftover configuration: " + pprint.pformat(config))
        return obj