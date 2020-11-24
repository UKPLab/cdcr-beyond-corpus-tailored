from typing import Dict

import numpy as np
from mosestokenizer import MosesDetokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from python import DOCUMENT_ID, TOKEN, SENTENCE_IDX
from python.handwritten_baseline import SENTENCE_EMBEDDINGS
from python.handwritten_baseline.pipeline.data.base import Dataset, BaselineDataProcessorStage


class SentenceBertEmbeddingFeaturePreparationStage(BaselineDataProcessorStage):
    """
    Computes Sentence-BERT embeddings for each document sentence. See https://github.com/UKPLab/sentence-transformers
    """

    def __init__(self, pos, config, config_global, logger):
        super(SentenceBertEmbeddingFeaturePreparationStage, self).__init__(pos, config, config_global, logger)

        self._pretrained_model_name = config["pretrained_model_name"]

        self._cache = self._provide_cache("sentence_bert", bind_parameters=config)

        # note: we are not using NLTK TreebankWordDetokenizer here, because that one replaces double quotes with two
        # single quotes which makes mappings between the tokenized and detokenized strings needlessly complicated
        self._detokenizer = MosesDetokenizer("en")

    def _process_dataset(self,
                         dataset: Dataset,
                         live_objects: Dict) -> Dataset:

        transformer = SentenceTransformer(self._pretrained_model_name)

        index = {}
        mat_embeddings = [] # List[np.array]

        # detokenize each sentence and run it through the sentence BERT transformer, unless it's already cached
        for (doc_id, sent_idx), df in tqdm(dataset.tokens.groupby([DOCUMENT_ID, SENTENCE_IDX]), desc="Obtaining sentence BERT embeddings", mininterval=10):
            detok_sentence = self._detokenizer(df[TOKEN].values.tolist())

            if not detok_sentence in self._cache:
                embedded_sentence = transformer.encode([detok_sentence], show_progress_bar=False, batch_size=1)[0]
                self._cache[detok_sentence] = embedded_sentence
            else:
                embedded_sentence = self._cache[detok_sentence]

            index[(doc_id, sent_idx)] = len(index)
            mat_embeddings.append(embedded_sentence.astype(np.float16))

        mat_embeddings = np.vstack(mat_embeddings)

        # and we're done
        sentence_embeddings = (index, mat_embeddings)
        dataset.set(SENTENCE_EMBEDDINGS, sentence_embeddings)

        return dataset


component = SentenceBertEmbeddingFeaturePreparationStage
