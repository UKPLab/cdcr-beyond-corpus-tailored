from typing import Tuple, Optional

import pandas as pd

from python import DOCUMENT_ID, MENTION_ID, SENTENCE_IDX, TOKEN_IDX_FROM
from python.handwritten_baseline import MENTION_TYPE_COARSE, COMPONENT_MENTION_ID


def look_up_event_component_by_srl(x: Tuple[pd.DataFrame, str, str],
                                   idx: Tuple,
                                   semantic_roles: pd.DataFrame
                                   ) -> Optional[pd.Series]:
    """
    Given an action mention, use semantic roles and more to find the most fitting row in the given event component
    dataframe. This can be used for example to find the temporal mention for an action.
    :param x: tuple of: mentions dataframe, the coarse mention type represented by the dataframe, and the column whose contents can be used as a hint to find the most precise mention inin case no semantic roles exist for an action phrase
    :param semantic_roles: semantic roles
    :param idx: (doc_id, mention_id) of the action mention
    :return: the pd.Series with the best fitting event component
    """
    mentions_df, mention_type_coarse, column_length_hint = x
    assert column_length_hint in mentions_df.columns

    assert len(idx) == 2
    doc_id, mention_id = idx

    # if there are no event component mentions at all in this document, bail out
    if not doc_id in mentions_df.index.get_level_values(DOCUMENT_ID):
        return None

    # find the arguments for this mention which the SRL system found
    document_matches = (semantic_roles[DOCUMENT_ID] == doc_id)
    mention_id_matches = (semantic_roles[MENTION_ID] == mention_id)
    mention_type_matches = (semantic_roles[MENTION_TYPE_COARSE] == mention_type_coarse)
    args = semantic_roles.loc[document_matches & mention_id_matches & mention_type_matches]
    mentions_of_args = mentions_df.loc[doc_id].merge(args,
                                                     left_index=True,
                                                     right_on=[COMPONENT_MENTION_ID])

    if not mentions_of_args.empty:
        # if there are multiple mentions, use the length of a certain column as a heuristic for finding the most precise event component mention
        return mentions_of_args.loc[(mentions_of_args[column_length_hint].str.len().idxmax())]

    return None


def look_up_event_component_by_sentence(x: Tuple[pd.DataFrame, str, str],
                                        idx: Tuple,
                                        mentions_action: pd.DataFrame) -> Optional[pd.Series]:
    """
    Given an action mention, use token distance to find the most fitting row in the given event component dataframe.
    This can be used for example to find the temporal mention for an action.
    :param x: tuple of: mentions dataframe, the coarse mention type represented by the dataframe, and the column whose contents can be used as a hint to find the most precise mention inin case no semantic roles exist for an action phrase
    :param mentions_action: action mentions
    :param idx: (doc_id, mention_id) of the action mention
    :return: the pd.Series with the best fitting event component
    """
    mentions_df, _, _ = x

    assert len(idx) == 2
    doc_id, mention_id = idx

    # if there are no event component mentions at all in this document, bail out
    if not doc_id in mentions_df.index.get_level_values(DOCUMENT_ID):
        return None
    mentions_event_component_in_doc = mentions_df.loc[doc_id]

    # try to return the closest event component mention from the same sentence
    action_mention = mentions_action.loc[idx]
    if action_mention[SENTENCE_IDX] in mentions_event_component_in_doc[SENTENCE_IDX].values:
        mentions_in_sentence = mentions_event_component_in_doc.loc[mentions_event_component_in_doc[SENTENCE_IDX] == action_mention[SENTENCE_IDX]]
        index_of_closest_mention = (mentions_in_sentence[TOKEN_IDX_FROM] - action_mention[TOKEN_IDX_FROM]).abs().argmin()
        return mentions_in_sentence.iloc[index_of_closest_mention]
    return None


def look_up_event_component_from_closest_preceding_sentence(x: Tuple[pd.DataFrame, str, str],
                                                            idx: Tuple,
                                                            mentions_action: pd.DataFrame) -> Optional[pd.Series]:
    """
    Given an action mention, find the row in the given event component dataframe which is closest to the mention from a
    preceding sentence in the document. This can be used for example to find the temporal mention for an action.
    :param x: tuple of: mentions dataframe, the coarse mention type represented by the dataframe, and the column whose contents can be used as a hint to find the most precise mention inin case no semantic roles exist for an action phrase
    :param mentions_action: action mentions
    :param idx: (doc_id, mention_id) of the action mention
    :return: the pd.Series with the best fitting event component
    """
    mentions_df, mention_type_coarse, column_length_hint = x
    assert column_length_hint in mentions_df.columns

    assert len(idx) == 2
    doc_id, mention_id = idx

    # if there are no event component mentions at all in this document, bail out
    if not doc_id in mentions_df.index.get_level_values(DOCUMENT_ID):
        return None

    # return an event component expression from the closest preceding sentences find event component mentions in
    # preceding sentences of the same document as the given mention
    mentions_event_component_in_doc = mentions_df.loc[doc_id]
    sent_idx_of_action = mentions_action.at[idx, SENTENCE_IDX]
    preceding_mentions_in_doc = mentions_event_component_in_doc.loc[mentions_event_component_in_doc[SENTENCE_IDX] < sent_idx_of_action]

    # if there are any, return the closest preceding one (highest sentence index)
    if not preceding_mentions_in_doc.empty:
        mentions_in_closest_sentence = preceding_mentions_in_doc.loc[preceding_mentions_in_doc[SENTENCE_IDX] == preceding_mentions_in_doc[SENTENCE_IDX].max()]
        # if there are multiple mentions, use the length of a certain column as a heuristic for finding the most precise event component mention
        return mentions_in_closest_sentence.loc[(mentions_in_closest_sentence[column_length_hint].str.len().idxmax())]
    return None


def look_up_document_level_event_component(mentions_df, idx) -> Optional[pd.Series]:
    """
    Return pd.Series of first event component phrase in a document (if there is one).
    :param mentions_df:
    :param idx: (doc_id, mention_id) of an action mention
    :return:
    """
    assert len(idx) == 2
    doc_id, mention_id = idx

    if not doc_id in mentions_df.index.get_level_values(DOCUMENT_ID):
        return None
    # mentions are sorted by occurrence in the document, hence we can simply take the first
    return mentions_df.loc[doc_id].iloc[0]