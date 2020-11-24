from pathlib import Path
from typing import Dict

import numpy as np
from allennlp.common.util import import_module_and_submodules
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from tqdm import tqdm

from python import SENTENCE_IDX, TOKEN, TOKEN_IDX_FROM, TOKEN_IDX_TO
from python.handwritten_baseline import ACTION_PHRASE_EMBEDDINGS
from python.handwritten_baseline.pipeline.data.base import Dataset, BaselineDataProcessorStage
from python.handwritten_baseline.pipeline.data.processing.feature_preparation.action_phrase.allennlp import \
    SpanEmbedderPredictor


class ActionPhraseEmbeddingFeaturePreparationStage(BaselineDataProcessorStage):
    """
    Computes embedded span representations for action phrases.
    """

    def __init__(self, pos, config, config_global, logger):
        super(ActionPhraseEmbeddingFeaturePreparationStage, self).__init__(pos, config, config_global, logger)

        # path to the handcrafted model.tar.gz archive file of the span embedder model
        span_embedder_model_path = Path(config["model"])
        assert span_embedder_model_path.exists() and span_embedder_model_path.is_file()

        # instantiate model with pretrained weights from the archive file
        archive = load_archive(span_embedder_model_path)

        # before instantiating our custom predictor with the archived model, we need to tell AllenNLP about the package where the predictor lives
        import_module_and_submodules("python.handwritten_baseline.pipeline.data.processing.feature_preparation.action_phrase.allennlp")
        self._span_embedder = Predictor.from_archive(archive, "span_embedder")  # type: SpanEmbedderPredictor

        self._cache = self._provide_cache("action_phrase_embeddings", bind_parameters=config)

    def _process_dataset(self,
                         dataset: Dataset,
                         live_objects: Dict) -> Dataset:
        mentions_action = dataset.mentions_action
        tokens = dataset.tokens

        index = {}
        mat_embeddings = [] # List[np.array]

        # determine sentences with action mentions
        for idx, row in tqdm(mentions_action.iterrows(),
                                                desc="Obtaining contextualized action phrase representations",
                                                mininterval=10,
                                                total=len(mentions_action)):
            assert len(idx) == 2
            doc_id, mention_id = idx

            doc = tokens.loc[doc_id]

            action_phrase = mentions_action.loc[(doc_id, mention_id)]
            sent_idx_of_action = action_phrase[SENTENCE_IDX]

            # take 2*n sentences surrounding this action phrase's sentence
            NUM_CONTEXT_SENTENCES = 2
            sent_from = max(0, sent_idx_of_action-NUM_CONTEXT_SENTENCES)
            sent_to = min(sent_idx_of_action+NUM_CONTEXT_SENTENCES, doc.index.get_level_values(SENTENCE_IDX).max())
            context = doc.loc[(slice(sent_from, sent_to))]

            # we need to compute the span of the action mention inside the context
            span_from = context.index.get_loc((sent_idx_of_action, action_phrase[TOKEN_IDX_FROM]))
            # span interval boundaries in our df are exclusive which means the end boundary of a span located at the
            # very end of a sentence would produce a KeyError here, circumvent that by subtracting, then adding 1 again
            span_to = context.index.get_loc((sent_idx_of_action, action_phrase[TOKEN_IDX_TO] - 1)) + 1

            # obtain embedded representation (from the cache, if available, otherwise compute it fresh)
            tokenized_context = context[TOKEN].values
            spans = [[span_from, span_to]]

            cache_key = " ".join(tokenized_context) + repr(spans)
            if cache_key not in self._cache:
                embedded_repr = self._span_embedder.predict_tokenized(tokenized_context, spans)
                self._cache[cache_key] = embedded_repr
            else:
                embedded_repr = self._cache[cache_key]

            # add embeddings to index
            index[idx] = len(index)
            mat_embeddings.append(embedded_repr)

        if mat_embeddings:
            mat_embeddings = np.vstack(mat_embeddings)
        else:
            mat_embeddings = None
        action_phrase_embeddings = (index, mat_embeddings)
        dataset.set(ACTION_PHRASE_EMBEDDINGS, action_phrase_embeddings)

        return dataset


component = ActionPhraseEmbeddingFeaturePreparationStage