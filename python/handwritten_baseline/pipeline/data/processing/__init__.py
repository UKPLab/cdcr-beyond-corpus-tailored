from typing import List

import numpy as np
import pandas as pd

from python import DOCUMENT_ID, MENTION_ID, SENTENCE_IDX, TOKEN_IDX_FROM, TOKEN_IDX_TO
from python.handwritten_baseline.pipeline.data.base import Dataset
from python.util.spans import span_matching


def left_join_predictions(anno_gold: pd.DataFrame, anno_predicted: pd.DataFrame, columns_keep_gold: List[str],
                          columns_keep_system: List[str]) -> pd.DataFrame:
    """
    Given gold mention annotations and predicted mention annotations, this method returns the gold annotations with
    additional columns from the system prediction merged in, based on the optimal 1:1 span matching per sentence. Gold
    annotation spans will not be modified, only enriched (hence: left join). Index and column of dataframes must
    conform to a certain format (see assert in code). Spans in the dataframes must be non-overlapping.
    :param anno_gold:
    :param anno_predicted:
    :param columns_keep_gold:
    :param columns_keep_system:
    :return:
    """
    assert anno_gold.index.names == [DOCUMENT_ID, MENTION_ID]
    assert anno_predicted.index.names == [DOCUMENT_ID, MENTION_ID]

    mappings = []
    MENTION_ID_GOLD = "mention-id-gold"
    MENTION_ID_PREDICTED = "mention-id-predicted"

    # perform intersection sentence-wise
    if not anno_predicted.empty:
        for (doc_id, sent_idx), df_gold in anno_gold.reset_index().groupby([DOCUMENT_ID, SENTENCE_IDX]):
            spans_gold = df_gold[[TOKEN_IDX_FROM, TOKEN_IDX_TO]].values.tolist()

            # look up mentions at the same spot in system output
            anno_predicted_wout_index = anno_predicted.reset_index()
            df_predicted = anno_predicted_wout_index.loc[(anno_predicted_wout_index[DOCUMENT_ID] == doc_id) & (anno_predicted_wout_index[SENTENCE_IDX] == sent_idx)]
            spans_predicted = df_predicted[[TOKEN_IDX_FROM, TOKEN_IDX_TO]].values.tolist()

            # perform span matching (only based on spans! no type information taken into consideration!)
            matched_spans = span_matching(spans_gold, spans_predicted, keep_A=True)

            # keep MENTION_IDs of matched mentions
            for i_gold, i_predicted in matched_spans.items():
                row = {DOCUMENT_ID: doc_id,
                       MENTION_ID_GOLD: df_gold.iloc[i_gold][MENTION_ID]}
                # this index can be None because we set keep_A=True for span_matching, to always keep all gold annotations
                if i_predicted is not None:
                    row[MENTION_ID_PREDICTED] = df_predicted.iloc[i_predicted][MENTION_ID]
                mappings.append(row)
    mappings = pd.DataFrame(mappings, columns=[DOCUMENT_ID, MENTION_ID_GOLD, MENTION_ID_PREDICTED])

    if not mappings.empty:
        # merge in the columns we want to keep from the gold annotations
        mappings = mappings.merge(anno_gold[columns_keep_gold],
                                  left_on=[DOCUMENT_ID, MENTION_ID_GOLD],
                                  right_index=True)
        # merge in the columns we want to keep from the predicted annotations - note the use of how="left" to keep gold annotations which have MENTION_ID_PREDICTED == None
        left_joined = mappings.merge(anno_predicted[columns_keep_system],
                                     left_on=[DOCUMENT_ID, MENTION_ID_PREDICTED],
                                     right_index=True,
                                     how="left")

        # drop unwanted columns, return to original column names, return to original index
        left_joined = left_joined.drop(columns=[MENTION_ID_PREDICTED])
        left_joined = left_joined.rename(columns={MENTION_ID_GOLD: MENTION_ID})
        left_joined = left_joined.set_index([DOCUMENT_ID, MENTION_ID])
    else:
        # append lots of NaNs if there is nothing to merge
        left_joined = pd.concat([anno_gold[columns_keep_gold], pd.DataFrame([], columns=columns_keep_system)], axis=1)
    left_joined.sort_index(inplace=True)
    return left_joined


def outer_join_predictions(predicted_entities_df: pd.DataFrame, dataset: Dataset) -> pd.DataFrame:
    """
    Given a dataframe of predicted mentions, this method searches through all gold mentions in a dataset, and returns
    a dataframe of all predicted mentions which do not overlap with gold mentions. The returned dataframe can be
    readily combined with the original dataset, i.e. mention IDs in the returned dataframe do not conflict with those
    in the gold data.
    :param predicted_entities_df:
    :param dataset:
    :return:
    """

    # We need to make sure any entities added to the dataset do not overlap with existing entities. Obtain
    # dataframe with spans of each entity known so far:
    gold_entities_with_spans = [dataset.mentions_action, dataset.mentions_participants, dataset.mentions_time, dataset.mentions_location]
    if dataset.mentions_other is not None:
        gold_entities_with_spans.append(dataset.mentions_other)
    gold_entities_with_spans = pd.concat(gold_entities_with_spans)[[SENTENCE_IDX, TOKEN_IDX_FROM, TOKEN_IDX_TO]].sort_index().reset_index()

    # When adding new entities, we need to assign mention IDs. Mention IDs are assigned (should be, at least)
    # sequentially in each document starting with 0. Find the lowest unused mention identifier in each document:
    max_mention_id_per_doc = gold_entities_with_spans.groupby(DOCUMENT_ID)[MENTION_ID].max()

    predicted_entities_to_add = []
    for doc_id, doc_entities in predicted_entities_df.groupby(DOCUMENT_ID):
        # determine ID to use for new entities
        next_mention_id = 0
        if doc_id in max_mention_id_per_doc:
            next_mention_id = max_mention_id_per_doc.at[doc_id] + 1

        # for each predicted mention, check if it overlaps with a gold mention, if not, keep it
        for idx, entity in doc_entities.iterrows():
            gold_entities_in_sent = gold_entities_with_spans.loc[(gold_entities_with_spans[DOCUMENT_ID] == doc_id) & (
                        gold_entities_with_spans[SENTENCE_IDX] == entity[SENTENCE_IDX])]

            if gold_entities_in_sent.empty:
                entity_span_is_overlap_free = True
            else:
                # perform intersection test with all gold entities
                max_span_from = np.maximum(entity[TOKEN_IDX_FROM], gold_entities_in_sent[TOKEN_IDX_FROM])
                min_span_to = np.minimum(entity[TOKEN_IDX_TO], gold_entities_in_sent[TOKEN_IDX_TO])
                entity_span_is_overlap_free = np.all(max_span_from >= min_span_to)

            if entity_span_is_overlap_free:
                predicted_entities_to_add.append({DOCUMENT_ID: doc_id, MENTION_ID: next_mention_id, **entity.to_dict()})
                next_mention_id += 1

    predicted_entities_to_add = pd.DataFrame(predicted_entities_to_add)
    if not predicted_entities_to_add.empty:
        predicted_entities_to_add.set_index([DOCUMENT_ID, MENTION_ID], inplace=True)
    return predicted_entities_to_add