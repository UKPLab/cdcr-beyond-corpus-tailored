from typing import Dict

import pandas as pd
from tqdm import tqdm

from python import DOCUMENT_ID, SENTENCE_IDX, TOKEN_IDX, PUBLISH_DATE, TOKEN, MENTION_ID, TOKEN_IDX_FROM, TOKEN_IDX_TO, \
    MENTION_TYPE, TIMEX_CORENLP_TO_OURS
from python.common_components import CORENLP
from python.common_components.corenlp import CoreNlp
from python.handwritten_baseline import POS, MENTION_TEXT, LEMMA, TIMEX_NORMALIZED
from python.handwritten_baseline.pipeline.data.base import BaselineDataProcessorStage, MODE_EXTEND, \
    MODE_INTERSECT, MODE_REPLACE
from python.handwritten_baseline.pipeline.data.base import Dataset
from python.handwritten_baseline.pipeline.data.processing import left_join_predictions, outer_join_predictions


class CoreNlpPosNerProcessorStage(BaselineDataProcessorStage):

    def __init__(self, pos, config, config_global, logger):
        super(CoreNlpPosNerProcessorStage, self).__init__(pos, config, config_global, logger)

        if self.mode == MODE_REPLACE:
            raise ValueError(f"Mode '{MODE_REPLACE}' is not supported.")

    @staticmethod
    def _parse_document(corenlp: CoreNlp,
                        doc_tokens: pd.DataFrame,
                        publish_date: pd.datetime):
        assert doc_tokens.index.names == [DOCUMENT_ID, SENTENCE_IDX, TOKEN_IDX]
        doc_id = doc_tokens.index.get_level_values(DOCUMENT_ID).unique()[0]

        tokens = []
        time_entities = []

        is_pretokenized = True

        # if present, pass the document publish date
        doc_date = None
        if pd.notna(publish_date):
            doc_date = publish_date.strftime("%Y-%m-%dT%H:%M:%S")

        for sent_idx, sent_tokens in doc_tokens.groupby(SENTENCE_IDX):
            properties = {}

            # The correct property to pass for the document date is "date", see https://stackoverflow.com/a/46290254.
            # When using any of the ner.docdate.xxx options, the CoreNLP server restarts for every request, which is terribly slow.
            if doc_date is not None:
                properties["date"] = doc_date
            if is_pretokenized:
                properties["tokenize.whitespace"] = True
                properties["tokenize.options"] = "ptb3Escaping=false"
                properties["ssplit.eolonly"] = True

            sentence_pretokenized = sent_tokens[TOKEN].str.cat(sep=" ")
            annotations = corenlp.parse_sentence(sentence_pretokenized, properties=properties)

            # if no "sentence" was recognized, keep the span as it was
            sentences = annotations.sentence
            if not sentences:
                raise ValueError

            # CoreNLP might tell us there is more than 1 sentence in the given string. We ignore that and trust our own
            # previously applied sentence tokenization. Therefore, tokens are counted across the corenlp sentence boundaries
            # here.
            sent_tokens = []
            token_idx = 0
            mentions = []
            sent_temporal_mentions = []

            for sentence in sentences:
                for token in sentence.token:
                    word, pos, lemma = token.word, token.pos, token.lemma
                    sent_tokens.append({TOKEN_IDX: token_idx, TOKEN: word, POS: pos, LEMMA: lemma})
                    token_idx += 1
                for mention in sentence.mentions:
                    _type, text = mention.entityType, mention.entityMentionText

                    row = {MENTION_TYPE: _type,
                           MENTION_TEXT: text,
                           TOKEN_IDX_FROM: int(mention.tokenStartInSentenceInclusive),
                           TOKEN_IDX_TO: int(mention.tokenEndInSentenceExclusive)}

                    # specific handling for temporal mentions
                    if _type in TIMEX_CORENLP_TO_OURS.keys():
                        # skip any non-timex mentions
                        if not mention.HasField("timex"):
                            continue

                        # update mention type and keep normalized TIMEX if we can get it
                        row[MENTION_TYPE] = TIMEX_CORENLP_TO_OURS[mention.timex.type]
                        if mention.HasField("normalizedNER"):
                            row[TIMEX_NORMALIZED] = mention.normalizedNER
                        sent_temporal_mentions.append(row)
                    else:
                        mentions.append(row)

            # add document id and sentence idx to each row
            for row in sent_tokens:
                row[DOCUMENT_ID] = doc_id
                row[SENTENCE_IDX] = sent_idx
            tokens += sent_tokens

            for row in sent_temporal_mentions:
                row[DOCUMENT_ID] = doc_id
                row[SENTENCE_IDX] = sent_idx
            time_entities += sent_temporal_mentions

        # combine tokens and mention info into new dataframes
        tokens = pd.DataFrame(tokens).set_index([DOCUMENT_ID, SENTENCE_IDX, TOKEN_IDX]).sort_index()

        # turn the index into [DOCUMENT_ID, MENTION_ID]
        if time_entities:
            time_entities_reindexed = []
            for doc_id, df in pd.DataFrame(time_entities).groupby(DOCUMENT_ID):
                temp = df.reset_index(drop=True).reset_index().rename(columns={"index": MENTION_ID})
                time_entities_reindexed.append(temp)
            time_entities = pd.concat(time_entities_reindexed).set_index([DOCUMENT_ID, MENTION_ID]).sort_index()
        else:
            time_entities = None

        return tokens, time_entities


    def _process_dataset(self,
                         dataset: Dataset,
                         live_objects: Dict) -> Dataset:
        corenlp = live_objects[CORENLP]

        # data is pre-tokenized, but run in through CoreNLP one more time for POS and named entities
        self.logger.info("Performing POS-tagging and NER")
        tokens = []
        time_entities = []

        for doc_id, doc_tokens in tqdm(dataset.tokens.groupby(level=[DOCUMENT_ID]),
                                                               desc="Parsing documents",
                                                               total=len(dataset.documents),
                                                               mininterval=10):
            # if present, pass the document publish date
            publish_date = pd.NaT
            if PUBLISH_DATE in dataset.documents.columns:
                publish_date = dataset.documents.loc[dataset.documents[DOCUMENT_ID] == doc_id, PUBLISH_DATE].item()

            doc_tokens, doc_time_entities = self._parse_document(corenlp, doc_tokens, publish_date)
            tokens.append(doc_tokens)
            if doc_time_entities is not None:
                time_entities.append(doc_time_entities)

        tokens = pd.concat(tokens).sort_index()
        time_entities = pd.concat(time_entities).sort_index()
        time_entities = time_entities.loc[time_entities[TIMEX_NORMALIZED].notna()]

        # add pos and lemma to each token in the dataset
        dataset.tokens = dataset.tokens.merge(tokens[[POS, LEMMA]], left_index=True, right_index=True)

        if dataset.mentions_time is None:
            dataset.mentions_time = time_entities
        else:
            # Common task for MODE_EXTEND and MODE_INTERSECT: perform left-join: keep all gold entity annotations,
            # but add information from CoreNLP output
            columns_keep_system = time_entities.columns.drop([TOKEN_IDX_FROM, TOKEN_IDX_TO, SENTENCE_IDX, MENTION_TYPE])
            dataset.mentions_time = left_join_predictions(dataset.mentions_time, time_entities, dataset.mentions_time.columns, columns_keep_system)

            if self.mode == MODE_INTERSECT:
                self.logger.info("Intersected new annotations with dataset from previous pipeline stages.")
            elif self.mode == MODE_EXTEND:
                self.logger.info("Extending set of temporal expressions in dataset with the ones found by CoreNLP")

                mentions_time_to_add = outer_join_predictions(time_entities, dataset).copy()
                dataset.mentions_time = pd.concat([dataset.mentions_time, mentions_time_to_add]).sort_index()

        # add mention text for later
        def get_mention_text_from_mention(row: pd.Series) -> str:
            return " ".join(dataset.tokens.loc[(row.name[0], row[SENTENCE_IDX], slice(row[TOKEN_IDX_FROM], row[TOKEN_IDX_TO] - 1)), TOKEN].values)
        dataset.mentions_time[MENTION_TEXT] = dataset.mentions_time.apply(get_mention_text_from_mention, axis=1)
        return dataset


component = CoreNlpPosNerProcessorStage