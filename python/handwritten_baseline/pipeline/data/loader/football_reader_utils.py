import pathlib
from typing import Optional, Tuple, List

import pandas as pd

from python import *


def read_split_data(data_dir: pathlib.Path, token_level_data_dir: Optional[pathlib.Path] = None) -> List[Tuple]:
    """
    Reads one split of the FCC.
    :param data_dir: directory with gen 1 FCC files: documents, tokens, mentions_cross_subtopic, mentions_seminal_other
    :param token_level_data_dir: directory with gen 2 FCC files: participants, times, etc.
    :return:
    """
    # find files
    documents_path, tokens_path, mentions_cross_subtopic, mentions_seminal_other, event_vocabs = None, None, None, None, {}
    for p in data_dir.rglob("*"):
        if p.name ==  "documents.csv":
            documents_path = p
        if p.name == "tokens.csv":
            tokens_path = p
        if p.name == "mentions_cross_subtopic.csv":
            mentions_cross_subtopic = p
        if p.name == "mentions_seminal_other.csv":
            mentions_seminal_other = p

        if p.suffix == ".yaml":
            event_vocabs[p.stem] = p

    # load corpus
    documents = pd.read_csv(documents_path, parse_dates=[PUBLISH_DATE])
    tokens = pd.read_csv(tokens_path, index_col=[0, 1, 2])
    mentions_cross_subtopic = pd.read_csv(mentions_cross_subtopic, index_col=[0, 1])
    mentions_seminal_other = pd.read_csv(mentions_seminal_other, index_col=[0, 1])

    # for documents, create multiindex of the form (topic, subtopic, document): any document with unknown seminal event
    # (== unknown subtopic) will be placed in a unique singleton subtopic containing only that document
    documents[SUBTOPIC] = documents[SEMINAL_EVENT].where(documents[SEMINAL_EVENT].notna(),
                                                         other="singleton_subtopic_" + documents[DOCUMENT_ID])
    documents[TOPIC_ID] = "football_tournaments"
    documents = documents.set_index([TOPIC_ID, SUBTOPIC, DOCUMENT_ID]).sort_index()
    # for reasons of legacy code we always have the doc-id as a column too
    documents[DOCUMENT_ID] = documents.index.get_level_values(DOCUMENT_ID)

    output = [(documents, tokens, mentions_cross_subtopic, mentions_seminal_other, None)]

    # load the token-level extension if available
    if token_level_data_dir is not None:
        cross_subtopic_mentions_action_path, cross_subtopic_mentions_location_path, cross_subtopic_mentions_participants_path, cross_subtopic_mentions_time_path, cross_subtopic_semantic_roles_path = None, None, None, None, None
        for p in token_level_data_dir.rglob("*"):
            if p.name == "cross_subtopic_mentions_action.csv":
                cross_subtopic_mentions_action_path = p
            if p.name == "cross_subtopic_mentions_participants.csv":
                cross_subtopic_mentions_participants_path = p
            if p.name == "cross_subtopic_mentions_time.csv":
                cross_subtopic_mentions_time_path = p
            if p.name == "cross_subtopic_mentions_location.csv":
                cross_subtopic_mentions_location_path = p
            if p.name == "cross_subtopic_semantic_roles.csv":
                cross_subtopic_semantic_roles_path = p

        cross_subtopic_mentions_action = pd.read_csv(cross_subtopic_mentions_action_path, index_col=[0, 1])
        cross_subtopic_mentions_participants = pd.read_csv(cross_subtopic_mentions_participants_path, index_col=[0, 1])
        cross_subtopic_mentions_time = pd.read_csv(cross_subtopic_mentions_time_path, index_col=[0, 1])
        cross_subtopic_mentions_location = pd.read_csv(cross_subtopic_mentions_location_path, index_col=[0, 1])
        cross_subtopic_semantic_roles = pd.read_csv(cross_subtopic_semantic_roles_path)

        output.append((cross_subtopic_mentions_action, cross_subtopic_mentions_participants,
                       cross_subtopic_mentions_time, cross_subtopic_mentions_location, cross_subtopic_semantic_roles))

    return output