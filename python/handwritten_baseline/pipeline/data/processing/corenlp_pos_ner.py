from typing import Dict

import pandas as pd
from tqdm import tqdm

from python import *
from python.common_components import CORENLP
from python.common_components.corenlp import CoreNlp
from python.handwritten_baseline import POS, MENTION_TEXT, LEMMA, TIMEX_NORMALIZED
from python.handwritten_baseline.pipeline.data.base import BaselineDataProcessorStage, MODE_EXTEND, \
    MODE_INTERSECT, MODE_REPLACE
from python.handwritten_baseline.pipeline.data.base import Dataset
from python.handwritten_baseline.pipeline.data.processing import left_join_predictions, outer_join_predictions


class CoreNlpProcessorStage(BaselineDataProcessorStage):

    def __init__(self, pos, config, config_global, logger):
        super(CoreNlpProcessorStage, self).__init__(pos, config, config_global, logger)

        if self.mode == MODE_REPLACE:
            raise ValueError(f"Mode '{MODE_REPLACE}' is not supported.")

        self._keep_constituency_trees = config.get("keep_constituency_trees", False)
        self._keep_dependency_trees = config.get("keep_dependency_trees", False)

    def _process_dataset(self,
                         dataset: Dataset,
                         live_objects: Dict) -> Dataset:
        corenlp = live_objects[CORENLP] # type: CoreNlp

        # data is pre-tokenized, but run in through CoreNLP one more time for POS
        self.logger.info("Annotating documents with CoreNLP")

        properties = {"annotators": "ssplit,tokenize,pos,lemma,parse",
                      "tokenize.whitespace": True,
                      "tokenize.options": "ptb3Escaping=false",
                      "ssplit.isOneSentence": True}

        # prepare lookup of document publication dates
        if PUBLISH_DATE in dataset.documents.columns:
            doc_dates = dataset.documents.loc[dataset.documents[PUBLISH_DATE].notna(), PUBLISH_DATE].reset_index().drop(columns=[TOPIC_ID, SUBTOPIC]).set_index(DOCUMENT_ID)[PUBLISH_DATE].map(lambda d: d.strftime("%Y-%m-%dT%H:%M:%S"))
            doc_dates = doc_dates.to_dict()
        else:
            doc_dates = {}

        def sentences_it():
            for _, sent_tokens in dataset.tokens.groupby([DOCUMENT_ID, SENTENCE_IDX]):
                yield sent_tokens[TOKEN].str.cat(sep=" ")
        def properties_it():
            for (doc_id, _), _ in dataset.tokens.groupby([DOCUMENT_ID, SENTENCE_IDX]):
                if doc_id in doc_dates:
                    doc_properties = properties.copy()
                    # The correct property to pass for the document date is "date", see https://stackoverflow.com/a/46290254.
                    # When using any of the ner.docdate.xxx options, the CoreNLP server restarts for every request, which is terribly slow.
                    doc_properties["date"] = doc_dates.get(doc_id, None)
                    yield doc_properties
                else:
                    yield properties

        tokens = []
        time_entities = []
        constituency_trees = []
        dependency_trees = []

        num_sentences = len(dataset.tokens.reset_index()[[DOCUMENT_ID,SENTENCE_IDX]].drop_duplicates())
        sent_indexes = dataset.tokens.index.to_frame(index=False).groupby([DOCUMENT_ID, SENTENCE_IDX])
        sent_annos = corenlp.parse_strings(sentences_it(), properties=properties_it(), use_cache=False)

        for ((doc_id, sent_idx), _), annotations in tqdm(zip(sent_indexes, sent_annos),
                                                         desc="Annotating sentences",
                                                         total=num_sentences,
                                                         mininterval=10):
            token_idx = 0
            if annotations is None:
                self.logger.warning(f"Failed annotating ({doc_id}, {sent_idx}).")
                continue

            for sentence in annotations.sentence:
                for token in sentence.token:
                    word, pos, lemma = token.word, token.pos, token.lemma
                    tokens.append({DOCUMENT_ID: doc_id, SENTENCE_IDX: sent_idx, TOKEN_IDX: token_idx, TOKEN: word, POS: pos, LEMMA: lemma})
                    token_idx += 1
                for mention in sentence.mentions:
                    _type, text = mention.entityType, mention.entityMentionText

                    # skip any non-temporal mentions or non-timex mentions or mentions without normalized TIMEX
                    if not _type in TIMEX_CORENLP_TO_OURS.keys() or not mention.HasField("timex") or not mention.HasField("normalizedNER"):
                        continue

                    # update mention type and keep normalized TIMEX if we can get it
                    time_entities.append({DOCUMENT_ID: doc_id,
                           SENTENCE_IDX: sent_idx,
                           MENTION_TYPE: TIMEX_CORENLP_TO_OURS[mention.timex.type],
                           MENTION_TEXT: text,
                           TOKEN_IDX_FROM: int(mention.tokenStartInSentenceInclusive),
                           TOKEN_IDX_TO: int(mention.tokenEndInSentenceExclusive),
                           TIMEX_NORMALIZED: mention.normalizedNER})

                if self._keep_constituency_trees:
                    assert hasattr(sentence, "parseTree")
                    tree_bytes = sentence.parseTree.SerializeToString()
                    constituency_trees.append(
                        {DOCUMENT_ID: doc_id, SENTENCE_IDX: sent_idx, CONSTITUENCY_TREE: tree_bytes})
                if self._keep_dependency_trees:
                    assert hasattr(sentence, "basicDependencies")
                    graph_bytes = sentence.basicDependencies.SerializeToString()
                    dependency_trees.append(
                        {DOCUMENT_ID: doc_id, SENTENCE_IDX: sent_idx, DEPENDENCY_TREE: graph_bytes})

        # Combine tokens into new dataframe and add POS and lemma to each token in the dataset. Use how="left" instead
        # of how="inner" (the default) to not lose original tokens whose annotation failed.
        tokens = pd.DataFrame(tokens).set_index([DOCUMENT_ID, SENTENCE_IDX, TOKEN_IDX]).sort_index()
        dataset.tokens = dataset.tokens.merge(tokens[[POS, LEMMA]], left_index=True, right_index=True, how="left")

        if time_entities:
            # turn the index into [DOCUMENT_ID, MENTION_ID]
            time_entities_reindexed = []
            for doc_id, df in pd.DataFrame(time_entities).groupby(DOCUMENT_ID):
                temp = df.reset_index(drop=True).reset_index().rename(columns={"index": MENTION_ID})
                time_entities_reindexed.append(temp)
            time_entities = pd.concat(time_entities_reindexed).set_index([DOCUMENT_ID, MENTION_ID]).sort_index()

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

        # add constituency and dependency parses to dataset object
        if self._keep_constituency_trees and constituency_trees:
            constituency_trees = pd.DataFrame(constituency_trees).set_index([DOCUMENT_ID, SENTENCE_IDX]).sort_index()
            constituency_trees.to_hdf(self.stage_disk_location / "constituency.hdf", key=CONSTITUENCY_TREE)
            dataset.set(CONSTITUENCY_TREE, constituency_trees)
        if self._keep_dependency_trees and dependency_trees:
            dependency_trees = pd.DataFrame(dependency_trees).set_index([DOCUMENT_ID, SENTENCE_IDX]).sort_index()
            dependency_trees.to_hdf(self.stage_disk_location / "dependency.hdf", key=DEPENDENCY_TREE)
            dataset.set(DEPENDENCY_TREE, dependency_trees)

        return dataset


component = CoreNlpProcessorStage