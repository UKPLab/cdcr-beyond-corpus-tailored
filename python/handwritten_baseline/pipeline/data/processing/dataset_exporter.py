import json
import pickle
from logging import Logger
from typing import Dict

import pandas as pd

from python import *
from python.handwritten_baseline.pipeline.data.base import Dataset, BaselineDataProcessorStage
from python.pipeline import ID, TIMESTAMP


class DatasetExporterStage(BaselineDataProcessorStage):
    """
    Writes a dataset to disk using pickle.
    """

    def __init__(self, pos, config, config_global, logger):
        super(DatasetExporterStage, self).__init__(pos, config, config_global, logger)

        self._filename_common_parts = [config["dataset_name"], config["split"], config_global[ID], config_global[TIMESTAMP]]
        self._do_export_barhom_format = config.get("export_in_barhom_format_as_well", False)
        self._do_export_cattan_format = config.get("export_in_cattan_format_as_well", False)
        self._do_export_plaintext_documents = config.get("export_plaintext_documents_as_well", False)

    def _export_dataset(self, dataset: Dataset):
        filename = "_".join(self._filename_common_parts[:2] + ["preprocessed"] + self._filename_common_parts[2:]) + ".pickle"
        output_file = self.stage_disk_location / filename

        with output_file.open("wb") as f:
            pickle.dump(dataset, f)

    def _export_cattan_format(self, dataset: Dataset):
        """
        Writes dataset to disk in the format used by the Cattan et al. 2020 ECB+ model.
        :param dataset:
        :return:
        """
        cattan_format_location = self.stage_disk_location / "cattan" / self.config["dataset_name"]
        cattan_format_location.mkdir(parents=True)
        mentions_file = cattan_format_location / (self.config["split"] + "_events.json")
        tokens_file = cattan_format_location / (self.config["split"] + ".json")
        conll_file = cattan_format_location / (self.config["split"] + "_corpus_level.conll")

        mentions, tokens = DatasetExporterStage._include_topic_subtopic_into_document_identifier(dataset, self.logger)

        # add a column "token-idx-global" which has the token index global per document
        tokens = tokens.groupby(DOCUMENT_ID).apply(
            lambda df: df.reset_index().reset_index().set_index([SENTENCE_IDX, TOKEN_IDX]).rename(
                columns={"index": "token-idx-global"}))
        tokens["token-idx-global"] += 1     # these are 1-based

        # map all token spans for all mentions to this global token indexing scheme
        mentions["token-length"] = mentions[TOKEN_IDX_TO] - mentions[TOKEN_IDX_FROM]
        mentions[TOKEN_IDX_FROM] = mentions[[DOCUMENT_ID, SENTENCE_IDX, TOKEN_IDX_FROM]].apply(lambda ser: tokens.loc[tuple(ser.values), "token-idx-global"], axis=1)
        mentions[TOKEN_IDX_TO] = mentions[TOKEN_IDX_FROM] + mentions["token-length"]
        mentions.drop(columns="token-length", inplace=True)

        # -------------------------------------------------------------------------------------------
        # (1) create mentions file
        # "{split}_{mention_type}.json" contain mentions
        # [
        #     {
        #         "doc_id": "2_11ecb.xml",
        #         "subtopic": "2_1",        <--- this is apparently never used, let's try getting away with not including it
        #         "m_id": "22",             <-- string type
        #         "sentence_id": "0",       <-- string type
        #         "tokens_ids": [           <--- list of int, globally 1-based
        #             38,
        #             39
        #         ],
        #         "tokens": "go on",        <-- never used, omit
        #         "tags": "",               <-- always empty string, omit
        #         "lemmas": "",             <-- always empty string, omit
        #         "cluster_id": 100000000,  <-- some uniquely identifying integer
        #         "cluster_desc": "",       <-- extra info which is never used, omit
        #         "topic": "2",             <-- string type
        #         "singleton": true
        #     },
        # ]
        def create_mentions_file(mentions):
            mentions["doc_id"] = mentions[DOCUMENT_ID]
            mentions["m_id"] = mentions[MENTION_ID].astype(str)
            mentions["sentence_id"] = mentions[SENTENCE_IDX].astype(str)
            mentions["tokens_ids"] = mentions.apply(lambda r: list(range(r[TOKEN_IDX_FROM], r[TOKEN_IDX_TO])), axis=1)
            mentions["cluster_id"] = mentions[EVENT].factorize()[0]
            mentions["topic"] = mentions[TOPIC_ID].astype(str)

            is_singleton = mentions[EVENT].value_counts().map(lambda v: v == 1)
            mentions["singleton"] = mentions[EVENT].map(is_singleton)

            mentions = mentions[["doc_id", "m_id", "sentence_id", "tokens_ids", "cluster_id", "topic", "singleton"]]
            with open(mentions_file, "w") as f:
                mentions.to_json(f, orient="records")
        create_mentions_file(mentions.copy())

        # -------------------------------------------------------------------------------------------

        # (2) create corpus file
        # "{split}.json" contain tokenized documents:
        # {
        #     doc_id: [
        #         [
        #             sent_idx,
        #             tok_idx,                                                  <--- 1-based and globally unique within a document
        #             token,
        #             boolean saying whether the sentence is in validated csv   <--- never used, just put "True"
        #         ],
        #         ...
        #     ],
        #     ...
        # }

        # mark all sentences for which we have action mentions as "validated" TODO not necessary, this is never used
        # sents_with_valid_mentions = mentions[[DOCUMENT_ID, SENTENCE_IDX]].drop_duplicates()
        # tokens["is-validated"] = tokens.index.to_series().map(lambda tup: tup[:2] in sents_with_valid_mentions.values)
        tokens["is-validated"] = True

        # strip every token - this has caused issues with Cattan et al. before, for example with the token "5\t\t" in
        # 20_2ecbplus.xml / DOC15646277841302609
        tokens[TOKEN] = tokens[TOKEN].str.strip()

        # remove empty tokens, like in 9_4ecbplus.xml / DOC15646106963482109
        tokens = tokens.loc[tokens[TOKEN].str.len() > 0]

        # write json iteratively, this is more efficient than collecting the to-be-written data structure in memory first
        with tokens_file.open("w") as f:
            f.write("{")
            for i, (doc_id, doc_df) in enumerate(tokens.drop(columns=DOCUMENT_ID).groupby(DOCUMENT_ID)):
                if i > 0:
                    f.write(",")  # comma for separating from previous document
                f.write(f'"{doc_id}":')
                tokens_as_list = doc_df.reset_index()[[SENTENCE_IDX, "token-idx-global", TOKEN, "is-validated"]].values.tolist()
                # minified, see https://stackoverflow.com/a/33233406
                json.dump(tokens_as_list, f, separators=(',', ':'))
            f.write("}")

        # -------------------------------------------------------------------------------------------

        # (3) create key CoNLL file
        # #begin document gun-violence
        # gun-violence	gun-violence_0	gun-violence_146565_104db82506283933234d28c49929a9cc	0	1	Shooting	True	(22)
        # gun-violence	gun-violence_0	gun-violence_146565_104db82506283933234d28c49929a9cc	0	2	at	True	-
        # gun-violence	gun-violence_0	gun-violence_146565_104db82506283933234d28c49929a9cc	0	3	Kokomo	True	-
        # gun-violence	gun-violence_0	gun-violence_146565_104db82506283933234d28c49929a9cc	0	4	bar	True	-
        # gun-violence	gun-violence_0	gun-violence_146565_104db82506283933234d28c49929a9cc	0	5	sends	True	-
        # ...

        # note: for Cattan et al., mention spans are unique (no stacked or nested mentions)
        mentions.sort_values(by=[TOPIC_ID, DOCUMENT_ID, SENTENCE_IDX, TOKEN_IDX_FROM], inplace=True)
        mentions["cluster-id"] = mentions[EVENT].factorize()[0]
        mentions_it = mentions.reset_index().iterrows()
        (_, next_mention) = next(mentions_it)

        # FIXME the subtopic column is not correct. Should be `gun-violence_0` but it's `gun-violence_12345`
        with conll_file.open("w") as f:
            f.write(f"#begin document {self.config['split']}_events\n")
            for _, topic_df in tokens.groupby(TOPIC_ID):
                for (doc_id, sent_idx, _), ser in topic_df.iterrows():
                    topic = doc_id.split("_")[0]
                    token_idx = ser["token-idx-global"]

                    # determine mention column
                    cluster_col = []
                    is_mention_in_same_sentence = next_mention is not None and next_mention[DOCUMENT_ID] == ser[DOCUMENT_ID] and next_mention[SENTENCE_IDX] == sent_idx
                    if is_mention_in_same_sentence:
                        is_mention_start = next_mention[TOKEN_IDX_FROM] == token_idx
                        is_in_mention =  next_mention[TOKEN_IDX_FROM] <= token_idx and next_mention[TOKEN_IDX_TO] > token_idx
                        is_mention_end = next_mention[TOKEN_IDX_TO] - 1 == token_idx

                        if is_mention_start:
                            cluster_col.append("(")
                        if is_in_mention:
                            cluster_col.append(str(next_mention["cluster-id"]))
                        if is_mention_end:
                            cluster_col.append(")")
                            # prepare next mention
                            try:
                                (_, next_mention) = next(mentions_it)
                            except StopIteration:
                                next_mention = None
                    cluster_col = "-" if not cluster_col else "".join(cluster_col)
                    line = [topic, f"{topic}_{ser[SUBTOPIC]}", ser[DOCUMENT_ID], str(sent_idx), str(token_idx), ser[TOKEN], str(ser["is-validated"]), cluster_col]

                    f.write("\t".join(line) + "\n")
            f.write("#end document")
        assert next_mention is None

    @staticmethod
    def _include_topic_subtopic_into_document_identifier(dataset: Dataset, logger: Logger):
        # To make the code of Barhom et al. work with different data without major changes, we need to put the
        # topic/subtopic information into the document identifier
        documents = dataset.documents
        documents.index = documents.index.droplevel(DOCUMENT_ID)
        mentions_action = dataset.mentions_action.reset_index().merge(documents.reset_index(), left_on=DOCUMENT_ID, right_on=DOCUMENT_ID)
        tokens = dataset.tokens.reset_index().merge(dataset.documents.reset_index(), on=DOCUMENT_ID)

        subtopic_column = SUBTOPIC
        if COLLECTION in documents.columns:
            logger.info("Found 'collection' column in documents dataframe (FCC corpus?). Dataset will be exported using 'collection' as a replacement for subtopics.")
            subtopic_column = COLLECTION

        get_conflated_doc_id = lambda df: df[TOPIC_ID].astype(str).str.replace("_", "-") + "_" + df[
            subtopic_column].astype(str).replace("_", "-") + "_" + df[DOCUMENT_ID].astype(str)
        mentions_action[DOCUMENT_ID] = get_conflated_doc_id(mentions_action)
        tokens[DOCUMENT_ID] = get_conflated_doc_id(tokens)
        tokens = tokens.set_index([DOCUMENT_ID, SENTENCE_IDX, TOKEN_IDX]).sort_index()
        return mentions_action, tokens

    def _export_barhom_format(self, dataset: Dataset):
        """
        Writes dataset to disk in the format used by the Barhom et al. 2019 ECB+ model.
        :param dataset:
        :return:
        """

        # prepare output files
        barhom_format_location = self.stage_disk_location / "barhom"
        barhom_format_location.mkdir()
        mentions_file = barhom_format_location / ("_".join(self._filename_common_parts + ["mentions"]) + ".json")
        corpus_file = barhom_format_location / ("_".join(self._filename_common_parts + ["corpus"]) + ".txt")
        corpus_doc_clustering_file = barhom_format_location / ("_".join(self._filename_common_parts + ["corpus_doc_clustering"]) + ".pkl")
        gold_doc_clustering_file = barhom_format_location / ("_".join(self._filename_common_parts + ["gold_transitive_closure_doc_clustering"]) + ".pkl")
        swirl_input_file_dir = barhom_format_location / "swirl_input"
        swirl_input_file_dir.mkdir()

        mentions_action, tokens = DatasetExporterStage._include_topic_subtopic_into_document_identifier(dataset, self.logger)

        # -------------------------------------------------------------------------------------------
        # (1) create mentions file
        def create_mentions_file(mentions):
            mentions = mentions[[DOCUMENT_ID, TOKEN_IDX_FROM, TOKEN_IDX_TO, SENTENCE_IDX, EVENT]]

            def get_mention_text_from_mention(row: pd.Series) -> str:
                return " ".join(tokens.loc[(row[DOCUMENT_ID], row[SENTENCE_IDX], slice(row[TOKEN_IDX_FROM], row[TOKEN_IDX_TO] - 1)), TOKEN].values)

            mentions["tokens_str"] = mentions.apply(get_mention_text_from_mention, axis=1)

            def get_mention_type(event: str):
                if event in MENTION_TYPES_ACTION_NEGATED:
                    return "NEG"
                return "ACT"

            mentions["mention_type"] = mentions[EVENT].astype(str).map(get_mention_type)

            is_singleton = mentions[EVENT].value_counts().map(lambda v: v == 1)
            mentions["is_singleton"] = mentions[EVENT].map(is_singleton)

            # convert to string, GVC has very large integer identifiers which JSON cannot represent
            mentions[EVENT] = mentions[EVENT].astype(str)

            mentions["score"] = -1.0
            mentions["is_continuous"] = True
            mentions["tokens_number"] = mentions.apply(
                lambda r: list(range(r[TOKEN_IDX_FROM], r[TOKEN_IDX_TO])), axis=1)

            mentions = mentions.reset_index()
            mentions.rename(columns={SENTENCE_IDX: "sent_id", DOCUMENT_ID: "doc_id", EVENT: "coref_chain"},
                            inplace=True)

            with open(mentions_file, "w") as f:
                mentions.to_json(f, orient="records", indent=4)
        create_mentions_file(mentions_action.copy())

        # -------------------------------------------------------------------------------------------

        # (2) create corpus file, format is one row per token, tab-delimited with columns:
        # doc_id, sent_id, token_num, word, coref_chain
        # 'coref_chain' is never used, we can just set it to "-" everywhere
        tokens[EVENT] = "-"
        tokens[TOKEN] = tokens[TOKEN].str.replace("\t", "")     # some tokens contain tabs which mess up the export in a moment
        with open(corpus_file, "w") as f:
            tokens.reset_index()[[DOCUMENT_ID, SENTENCE_IDX, TOKEN_IDX, TOKEN, EVENT]].to_csv(f, sep="\t", header=False, index=False)

        # -------------------------------------------------------------------------------------------

        # (3) create pickle file with all document texts, this is used for the document clustering at test time
        full_documents_as_dict = tokens[TOKEN].groupby(DOCUMENT_ID).apply(lambda ser: " ".join(ser.values)).to_dict()
        with open(corpus_doc_clustering_file, "wb") as f:
            pickle.dump(full_documents_as_dict, f)

        # -------------------------------------------------------------------------------------------

        # (4) determine transitive closure event coreference relation w.r.t. documents to create *the* gold preclustering
        mentions_and_documents = mentions_action[[DOCUMENT_ID, EVENT]].drop_duplicates()

        gold_doc_clustering = []
        for event, docs in mentions_and_documents.groupby(EVENT):
            docs_for_event = set(docs[DOCUMENT_ID].values)

            indices_of_intersecting_clusters = []
            for idx, doc_cluster in enumerate(gold_doc_clustering):
                if len(doc_cluster & docs_for_event) > 0:
                    indices_of_intersecting_clusters.append(idx)

            # get rid of old clusters
            new_cluster = set()
            for k, idx in enumerate(indices_of_intersecting_clusters):
                doc_cluster = gold_doc_clustering.pop(idx - k)
                new_cluster |= doc_cluster

            new_cluster |= docs_for_event
            gold_doc_clustering.append(new_cluster)

        # convert the partitioning in the usual list-of-lists format
        gold_doc_clustering = [list(doc_cluster) for doc_cluster in gold_doc_clustering]
        with open(gold_doc_clustering_file, "wb") as f:
            pickle.dump(gold_doc_clustering, f)


        # -------------------------------------------------------------------------------------------

        # (5) create input documents for SwiRL SRL system. See https://github.com/jcklie/SwiRL for working instructions
        # on how to set that up.
        # Yang et al. 2015 used input format (d), i.e. "3 (word)+" which does not use POS tagging or NER. Their input
        # files have filenames like SWIRL_INPUT.<doc-id> and the output files SWIRL_OUTPUT.<doc-id>. Barhom et al. 2019
        # do not report which SwiRL input format they used, but going from the fact that the SwiRL outputs in the github
        # repo start with SWIRL_OUTPUT, the next best assumption is that they used the same setup as Yang et al. 2015.
        # So do we then. Also, we mimic the file name pattern used by Barhom et al. so we don't need to change that
        # code as much...
        for doc_id, doc_tokens in tokens[TOKEN].groupby(DOCUMENT_ID):
            p = swirl_input_file_dir / ("SWIRL_INPUT." + doc_id + ".notxml.txt")

            # escape double quotes
            doc_tokens_escaped = doc_tokens.map(lambda tok: tok.replace('"', '\\"'))

            with p.open("w") as f:
                for _, sent_tokens_escaped in doc_tokens_escaped.groupby(SENTENCE_IDX):
                    f.write("3 " + " ".join(sent_tokens_escaped.values) + "\n")

    def _export_plaintext_documents(self, dataset):
        plaintext_dir = self.stage_disk_location / "plaintext_documents"
        plaintext_dir.mkdir(parents=True)

        for doc_id, tokens in dataset.tokens.groupby(DOCUMENT_ID)[TOKEN]:
            doc_text = " ".join(tokens.values)
            with (plaintext_dir / f"{doc_id}.txt").open("w") as f:
                f.write(doc_text)

    def _process_dataset(self,
                         dataset: Dataset,
                         live_objects: Dict) -> Dataset:
        # final assertion before exporting: make sure that all documents contain at least one action mention (this
        # broke our neck several times)
        documents_set = set(dataset.documents[DOCUMENT_ID].unique())
        mentions_set = set(dataset.mentions_action.index.get_level_values(DOCUMENT_ID).unique())
        assert len(documents_set) == len(documents_set & mentions_set)

        self._export_dataset(dataset)

        if self._do_export_barhom_format:
            self._export_barhom_format(dataset)

        if self._do_export_cattan_format:
            self._export_cattan_format(dataset)

        if self._do_export_plaintext_documents:
            self._export_plaintext_documents(dataset)

        return dataset


component = DatasetExporterStage
