from typing import List, Dict, Optional

import numpy as np
import torch
from allennlp.data import Field, Instance, DatasetReader, TokenIndexer, Token
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.data.fields import SpanField, TextField, ListField
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from allennlp.nn import InitializerApplicator
from allennlp.nn import util as nn_util
from allennlp.predictors.predictor import Predictor


# The classes in this module aid in producing span embeddings for tokens in a tokenized sentence.

@DatasetReader.register("span_embedder")
class SpanEmbedderDatasetReader(DatasetReader):

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 cache_directory: Optional[str] = None,
                 max_instances: Optional[int] = None) -> None:
        super().__init__(lazy, cache_directory, max_instances)

        self._token_indexers = token_indexers

    def text_to_instance(self, tokenized_sentence: List[str], spans: List[List[int]]) -> Instance:
        allennlp_sentence_tokens = [Token(text=t) for t in tokenized_sentence]
        sentence_token_indexes = TextField(allennlp_sentence_tokens, self._token_indexers)

        span_fields = []
        for span_start, span_end_exclusive in spans:
            span_field = SpanField(span_start, span_end_exclusive - 1, sentence_token_indexes)
            span_fields.append(span_field)

        fields: Dict[str, Field] = {}
        fields["tokens"] = sentence_token_indexes
        fields["spans"] = ListField(span_fields)
        return Instance(fields)


@Model.register("span_embedder")
class SpanEmbedder(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 span_extractor: SpanExtractor = None,
                 initializer: InitializerApplicator = InitializerApplicator()):
        """

        :param vocab: mandatory parameter
        :param text_field_embedder: embedder (BERT, LSTM, count-based, ...)
        :param span_extractor: Optional technique for extracting the span from a token sequence. If not provided, it is
                               assumed that `text_field_embedder` already returns a vectorized representation of each
                               instance, i.e. its output shape is `(batch_size, embed_dim)`.
        :param initializer:
        """
        super().__init__(vocab)
        self._text_field_embedder = text_field_embedder
        self._span_extractor = span_extractor

        initializer(self)

    def forward(self,
                tokens: TextFieldTensors,
                spans: torch.LongTensor) -> Dict[str, torch.Tensor]:
        # the shape of 'embedded' will be one of the following:
        #   (batch_size, max_input_sequence_length, embed_dim)
        #   (batch_size, embed_dim)
        embedded = self._text_field_embedder(tokens)

        assert len(embedded.shape) == 3

        # shape: (batch_size, max_input_sequence_length)
        source_mask = nn_util.get_text_field_mask(tokens)

        # shape: (batch, num_spans, embed_dim)
        span_embeddings = self._span_extractor(embedded, spans, sequence_mask=source_mask)

        return {"span_embeddings": span_embeddings}


@Predictor.register("span_embedder")
class SpanEmbedderPredictor(Predictor):

    def predict_tokenized(self, tokenized_sentence: List[str], spans: List[List[int]]) -> np.array:
        instance = self._dataset_reader.text_to_instance(tokenized_sentence, spans)
        json_dict = self.predict_instance(instance)

        span_embeddings = np.array(json_dict["span_embeddings"], dtype=np.float16)
        return span_embeddings