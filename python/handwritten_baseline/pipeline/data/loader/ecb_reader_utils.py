import logging
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Tuple, Optional

import pandas as pd

from python import TOPIC_ID, SUBTOPIC, DOCUMENT_NUMBER, DOCUMENT_ID, SENTENCE_IDX, TOKEN_IDX, TOKEN_IDX_TO, \
    TOKEN_IDX_FROM, TOKEN, MENTION_ID, EVENT, MENTION_TYPE, DESCRIPTION, MENTION_TYPES_ACTION

logger = logging.getLogger()


def read_xml(xml_path) -> Tuple[Any, Any, Any, Any, Any]:
    tree = ET.parse(xml_path)

    # 1: read document info
    root = tree.getroot()
    assert root.tag == "Document"
    doc_filename = root.attrib["doc_name"]
    doc_id = root.attrib["doc_id"]
    m = re.match(r"(?P<topic_id>\d+)_(?P<document_number>\d+)(?P<subtopic>\w+)\.xml", doc_filename)

    topic_id = m.group("topic_id")
    subtopic = m.group("subtopic")
    document_number = int(m.group("document_number"))

    documents_index = pd.MultiIndex.from_tuples([(topic_id, subtopic, doc_id)],
                                                names=[TOPIC_ID, SUBTOPIC, DOCUMENT_ID])
    documents = pd.DataFrame({DOCUMENT_ID: pd.Series(doc_id, index=documents_index),
                              DOCUMENT_NUMBER: pd.Series(document_number, index=documents_index)})

    # 2: read document content
    contents_rows = []
    contents_index = []
    for token_elmt in root.iter("token"):
        # index content
        sentence_idx = int(token_elmt.attrib["sentence"])
        token_idx = int(token_elmt.attrib["number"])
        contents_index.append((doc_id, sentence_idx, token_idx))

        # content
        token = token_elmt.text
        contents_rows.append({TOKEN: token})
    contents_index = pd.MultiIndex.from_tuples(contents_index, names=[DOCUMENT_ID, SENTENCE_IDX, TOKEN_IDX])
    contents = pd.DataFrame(contents_rows, index=contents_index)

    # 3: read markables / mentions and entity/event descriptions
    mentions_rows = []
    mentions_index = []
    entities_events = []
    for markable in root.find("Markables").getchildren():
        # Don't know what this is, skip it
        if markable.tag == "UNKNOWN_INSTANCE_TAG":
            continue

        mention_id = int(markable.attrib["m_id"])

        # there are markables without spans, these are descriptions of entities / events which we want to keep
        if "TAG_DESCRIPTOR" in markable.attrib.keys():
            if "instance_id" in markable.attrib.keys():
                entities_events.append({
                    EVENT: markable.attrib["instance_id"],
                    DESCRIPTION: markable.attrib["TAG_DESCRIPTOR"]
                })
            continue

        token_ids = [int(anchor.attrib["t_id"]) for anchor in markable.iter("token_anchor")]
        token_ids_from, token_ids_to = min(token_ids), max(token_ids)

        # the token_ids are cumulative token indexes, remove their cumulative nature
        token_indexes = contents.index.get_level_values(TOKEN_IDX).values
        token_idx_from = token_indexes[
            token_ids_from - 1]  # -1 because token_ids start at 1, so we need to access index 0 in the dataframe to find t_id 1
        token_idx_to = token_indexes[
                           token_ids_to - 1] + 1  # additionally +1 here because we want mention spans represented as intervals [from, to[

        sentence_idx = contents.index.get_level_values(SENTENCE_IDX).values[token_ids_from - 1]

        # resolve non-contiguous mentions
        is_non_contiguous_mention = len(token_ids) < token_idx_from - token_idx_to
        if is_non_contiguous_mention:
            logger.info("Converted non-contiguous mention to contiguous mention.")

        mentions_index.append((doc_id, mention_id))
        mentions_rows.append({SENTENCE_IDX: sentence_idx,
                              TOKEN_IDX_FROM: token_idx_from,
                              TOKEN_IDX_TO: token_idx_to,
                              MENTION_TYPE: markable.tag})
    mentions_index = pd.MultiIndex.from_tuples(mentions_index, names=[DOCUMENT_ID, MENTION_ID])
    mentions = pd.DataFrame(mentions_rows, index=mentions_index)
    entities_events = pd.DataFrame(entities_events).set_index(EVENT)

    # 4. read relations (clusters)
    clusters_rows = []
    for relation in root.find("Relations").getchildren():
        tags_of_interest = ["CROSS_DOC_COREF", "INTRA_DOC_COREF"]
        if not relation.tag in tags_of_interest:
            logger.info("Unexpected tag " + relation.tag)
            raise NotImplementedError

        # There are relations with tags INTRA_DOC_COREF and CROSS_DOC_COREF. The cross-doc ones have a "note" attribute.
        if "note" in relation.attrib:
            # this is the case for CROSS_DOC_COREF tags
            relation_id = relation.attrib["note"]
        else:
            # this is the case for INTRA_DOC_COREF tags
            relation_id = doc_id + "_" + relation.attrib["r_id"]

        for mention in relation.iter("source"):
            mention_id = int(mention.attrib["m_id"])
            clusters_rows.append({EVENT: relation_id, DOCUMENT_ID: doc_id, MENTION_ID: mention_id})
    clusters = pd.DataFrame(clusters_rows)

    # 5. create relations for singletons
    # In ECB plus, there are ACTION_OCCURRENCE markables which are not assigned to a relation. These are singletons. We
    # add one entry for each singleton to `clusters` to ensure consistency. Note that the opposite also exists:
    # singleton mentions which are marked as participating in a cross-doc coref relation, but there is no second
    # mention for this relation.
    if clusters.empty:
        singletons = mentions.index.to_frame().reset_index(drop=True)
    else:
        # This can most likely be done in a nicer way using some index difference...
        outer = pd.merge(mentions, clusters, left_index=True, right_on=[DOCUMENT_ID, MENTION_ID], how="outer")
        singletons = outer.loc[outer[EVENT].isna(), [DOCUMENT_ID, MENTION_ID]]
    singletons[EVENT] = "SINGLETON_" + singletons.astype(str).apply("_".join, axis=1)
    clusters = clusters.append(singletons, sort=False).reset_index(drop=True)

    return documents, contents, mentions, clusters, entities_events


def read_split_data(root: Path, sentence_filter_csv: Optional[Path]):
    documents = []
    contents = []
    mentions = []
    clusters = []
    entities_events = []

    # enumerate files
    for root, dirs, files in os.walk(str(root.absolute())):
        for file in files:
            path = os.path.abspath(os.path.join(root, file))
            f_documents, f_contents, f_mentions, f_clusters, f_entities_events = read_xml(path)

            documents.append(f_documents)
            contents.append(f_contents)
            mentions.append(f_mentions)
            clusters.append(f_clusters)
            entities_events.append(f_entities_events)

    documents = pd.concat(documents).sort_index()
    contents = pd.concat(contents).sort_index()
    mentions = pd.concat(mentions).sort_index()
    clusters = pd.concat(clusters, sort=False)
    entities_events = pd.concat(entities_events).sort_index()

    # assert that every mention participates only in one cluster -> meaning we can just add an 'EVENT' column to each mention
    assert clusters.duplicated(subset=[DOCUMENT_ID, MENTION_ID]).value_counts().get(True, 0) == 0

    clusters = clusters.set_index([DOCUMENT_ID, MENTION_ID])
    mentions = pd.merge(mentions, clusters, left_index=True, right_index=True).sort_index()

    # read file which tells us from which sentences we should keep event mentions
    if sentence_filter_csv is not None:
        sent_filter = pd.read_csv(sentence_filter_csv)
        doc_number_and_subtopic = sent_filter["File"].str.split("ecb", expand=True)
        doc_number_and_subtopic.columns = [DOCUMENT_NUMBER, SUBTOPIC]
        doc_number_and_subtopic[DOCUMENT_NUMBER] = doc_number_and_subtopic[DOCUMENT_NUMBER].astype(int)
        doc_number_and_subtopic[SUBTOPIC].replace({"plus": "ecbplus", "": "ecb"}, inplace=True)
        sent_filter = pd.concat([sent_filter.drop(columns="File"), doc_number_and_subtopic], axis=1)
        sent_filter.rename(columns={"Topic": TOPIC_ID, "Sentence Number": SENTENCE_IDX}, inplace=True)
        sent_filter[TOPIC_ID] = sent_filter[TOPIC_ID].astype(str)
        sent_filter = sent_filter[[TOPIC_ID, SUBTOPIC, DOCUMENT_NUMBER, SENTENCE_IDX]]

        # the sentence filter file applies to all splits, remove those topics that we don't have in the split we're loading
        topics_in_split = documents.index.get_level_values(TOPIC_ID).unique()
        sent_filter = sent_filter.loc[sent_filter[TOPIC_ID].isin(topics_in_split)].copy()

        # obtain doc-id from topic+subtopic+document number
        documents_with_doc_number_in_index = documents.set_index(DOCUMENT_NUMBER, append=True).reset_index(level=DOCUMENT_ID, drop=True).sort_index()
        sent_filter[DOCUMENT_ID] = sent_filter[[TOPIC_ID, SUBTOPIC, DOCUMENT_NUMBER]].apply(lambda row: documents_with_doc_number_in_index[DOCUMENT_ID].loc[tuple(row.values)], axis=1)

        all_mentions_to_keep = []
        for doc_id, df in mentions.groupby(DOCUMENT_ID):
            sentences_to_keep = sent_filter.loc[sent_filter[DOCUMENT_ID] == doc_id]

            # we only remove action phrases and leave the other mentions in place, so that we can potentially mask them for
            # analysis, see python.handwritten_baseline.pipeline.data.processing.masking.MentionMaskingStage
            is_official_evaluation_sentence = df[SENTENCE_IDX].isin(sentences_to_keep[SENTENCE_IDX])
            is_action_mention = df[MENTION_TYPE].isin(MENTION_TYPES_ACTION)
            mentions_to_keep = df.loc[is_official_evaluation_sentence | (~is_action_mention)]
            all_mentions_to_keep.append(mentions_to_keep)
        mentions = pd.concat(all_mentions_to_keep).sort_index()

    return documents, contents, mentions, entities_events