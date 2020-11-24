import textwrap
from pathlib import Path
from typing import List

import pandas as pd
from tabulate import tabulate

from python import SENTENCE_IDX, TOKEN, TOKEN_IDX_FROM, TOKEN_IDX_TO, DOCUMENT_ID, TOPIC_ID, SUBTOPIC
from python.handwritten_baseline import PREDICTION, LABEL, INSTANCE, IDX_A_MENTION, IDX_B_MENTION, IDX_A_DOC, IDX_B_DOC, \
    RECALL, PRECISION, F1
from python.handwritten_baseline.pipeline.data.base import Dataset
from python.util.util import get_dict_hash

PAIR_TYPE = "pair-type"
CT = "cross-topic"
CS = "cross-subtopic"
WS = "within-subtopic"
WD=  "within-document"
QUADRANT = "quadrant"
TP = "TP"
FP = "FP"
FN = "FN"
TN = "TN"

def perform_prediction_analysis(dataset: Dataset,
                                outcomes: List[pd.DataFrame],
                                num_samples_per_quadrant: int,
                                serialization_dir: Path) -> None:
    """
    Given outcomes from mention pair classifications, computes detailed confusion matrices per link type. Also picks one
    run and samples several instances for each quadrant of the 2x2 confusion matrix and prints those for manual analysis.
    :param dataset: evaluation dataset
    :param outcomes: list of dataframe containing evaluated pairs with predicted and gold label, one for each run
    :param num_samples_per_quadrant: number of instances sampled per confusion matrix quadrant
    :param serialization_dir
    :return:
    """

    # assert that all passed outcome dataframes are compatible
    df_lengths = [len(df) for df in outcomes]
    assert len(set(df_lengths)) == 1

    # check sameness of a-doc-ids and b-mention-ids, if one of those two mismatches we have a problem anyway
    a_doc_id_hashes = [get_dict_hash(df[IDX_A_DOC].values) for df in outcomes]
    b_mention_id_hashes = [get_dict_hash(df[IDX_B_MENTION].values) for df in outcomes]
    assert len(set(a_doc_id_hashes)) == 1
    assert len(set(b_mention_id_hashes)) == 1

    # All dataframes contain the same mention indices of each mention. We just need to keep this once, then we can throw
    # away mention indices for the outcomes of each run.
    index_df = outcomes[0][[IDX_A_DOC, IDX_A_MENTION, IDX_B_DOC, IDX_B_MENTION]].copy()
    for outcomes_df in outcomes:
        outcomes_df.drop(columns=[IDX_A_DOC, IDX_A_MENTION, IDX_B_DOC, IDX_B_MENTION], inplace=True)

    # In the mention pair index dataframe, label each pair with its type: cross-topic, cross-subtopic,
    # within-subtopic, within-document.
    # First, convert docs to usable format:
    docs = dataset.documents
    docs = pd.concat([docs.index.to_frame()[[TOPIC_ID, SUBTOPIC]].reset_index(drop=True), docs[DOCUMENT_ID].reset_index(drop=True)], axis=1)

    # Merging resets the index to the default. We want to keep it intact, so that we can concat index_df and the
    # outcomes again later.
    index_df_index = index_df.index
    index_df = index_df.merge(docs, left_on=IDX_A_DOC, right_on=DOCUMENT_ID, how="left")
    index_df = index_df.drop(columns=[DOCUMENT_ID]).rename(columns={TOPIC_ID: "a-topic-id", SUBTOPIC: "a-subtopic"})
    index_df = index_df.merge(docs, left_on=IDX_B_DOC, right_on=DOCUMENT_ID, how="left")
    index_df = index_df.drop(columns=[DOCUMENT_ID]).rename(columns={TOPIC_ID: "b-topic-id", SUBTOPIC: "b-subtopic"})
    index_df.index = index_df_index

    topic_match = (index_df["a-topic-id"] == index_df["b-topic-id"])
    subtopic_match = (index_df["a-subtopic"] == index_df["b-subtopic"])
    document_match = (index_df[IDX_A_DOC] == index_df[IDX_B_DOC])
    index_df.loc[~topic_match, PAIR_TYPE] = CT
    index_df.loc[topic_match & ~subtopic_match, PAIR_TYPE] = CS
    index_df.loc[topic_match & subtopic_match & ~document_match, PAIR_TYPE] = WS
    index_df.loc[topic_match & subtopic_match & document_match, PAIR_TYPE] = WD

    # For each run, label each pair with true positive, false positive, etc.
    for outcome_df in outcomes:
        outcome_df.loc[ outcome_df[LABEL] &  outcome_df[PREDICTION], QUADRANT] = TP
        outcome_df.loc[ outcome_df[LABEL] & ~outcome_df[PREDICTION], QUADRANT] = FN
        outcome_df.loc[~outcome_df[LABEL] &  outcome_df[PREDICTION], QUADRANT] = FP
        outcome_df.loc[~outcome_df[LABEL] & ~outcome_df[PREDICTION], QUADRANT] = TN

    _create_confusion_matrices(index_df, outcomes, serialization_dir)
    _print_prediction_pairs(index_df, outcomes[0], dataset, num_samples_per_quadrant, serialization_dir)

def _create_confusion_matrices(index_df: pd.DataFrame, outcomes: List[pd.DataFrame], serialization_dir: Path):
    out_dir = serialization_dir / "detailed_metrics"
    out_dir.mkdir(exist_ok=True, parents=True)

    RUN = "run"
    NUM_CASES = "num-cases"

    # for each run, obtain number of TP/FP/FN/TN for each type of link
    records = []
    for i, outcome_df in enumerate(outcomes):
        for link_type in [WD, WS, CS, CT]:
            outcomes_of_link_type = outcome_df.loc[index_df[PAIR_TYPE] == link_type]
            for quadrant in [TP, FP, FN, TN]:
                num_of_type_and_quadrant = (outcomes_of_link_type[QUADRANT] == quadrant).sum()
                records.append({RUN: i, PAIR_TYPE: link_type, QUADRANT: quadrant, NUM_CASES: num_of_type_and_quadrant})
    records = pd.DataFrame(records)

    # save records for later
    records.to_pickle(str(out_dir / "raw_confusion_matrix_records_df.pkl"))

    def compute_p_r_f1(df_with_quadrants: pd.DataFrame):
        """
        Given a dataframe with a multiindex in which QUADRANT is the deepest level, groups by the highest n-1 levels and
        computes recall, precision, F1 for each group.
        :param df_with_quadrants:
        :return:
        """
        index_cols = df_with_quadrants.index.names
        assert QUADRANT == index_cols[-1]

        index_columns_group = index_cols[:-1]

        metric_records = []
        for idx, df in df_with_quadrants.groupby(index_columns_group):
            # make sure index is list-typed
            if type(idx) is not tuple:
                idx = (idx,)

            assert len(df) == 4
            tp = df.xs(TP, level=-1).item()
            fp = df.xs(FP, level=-1).item()
            fn = df.xs(FN, level=-1).item()

            precision_denom = tp + fp
            precision = 0 if precision_denom == 0 else tp / precision_denom
            recall_denom = tp + fn
            recall = 0 if recall_denom == 0 else tp / recall_denom
            f1 = 0 if recall == 0 and precision == 0 else 2 * (precision * recall) / (precision + recall)

            record = {k:v for k, v in zip(index_columns_group, idx)}
            record[RECALL] = recall
            record[PRECISION] = precision
            record[F1] = f1
            metric_records.append(record)

        metrics = pd.DataFrame(metric_records).set_index(index_columns_group)
        return metrics

    def aggregate_metrics(df, axis, level=None):
        return df.describe(percentiles=[]).drop(["count", "50%"], axis=axis, level=level)

    # compute mean P, R, F1 over all runs, ignoring link types (this is what we already return from the scorer)
    metrics_by_run = compute_p_r_f1(records.groupby([RUN, QUADRANT])[NUM_CASES].sum())
    metrics_by_run_aggregated = aggregate_metrics(metrics_by_run, axis="index")
    with (out_dir / "p_r_f1_average_over_runs_ignoring_link_types.txt").open("w") as f:
        f.write("Mean P, R, F1 over all runs, ignoring coref link types (this is what we already return from the scorer)\n\n")
        f.write(tabulate(metrics_by_run_aggregated, headers="keys"))
    metrics_by_run_aggregated.to_pickle(out_dir / "p_r_f1_average_over_runs_ignoring_link_types.pkl")

    # compute mean P, R, F1 over all runs, but for each link type separately
    metrics_by_run_and_pair = compute_p_r_f1(records.groupby([RUN, PAIR_TYPE, QUADRANT])[NUM_CASES].sum())
    metrics_by_run_and_pair_aggregated = aggregate_metrics(metrics_by_run_and_pair.groupby(PAIR_TYPE), axis="columns", level=-1)
    with (out_dir / "p_r_f1_average_over_runs_for_each_link_type.txt").open("w") as f:
        f.write("Mean P, R, F1 over all runs, but for each link type separately\n\n")
        f.write(tabulate(metrics_by_run_and_pair_aggregated, headers="keys"))
    metrics_by_run_and_pair_aggregated.to_pickle(out_dir / "p_r_f1_average_over_runs_for_each_link_type.pkl")

    # compute mean absolute number of TP, FP, ... of each link type over all runs
    mean_absolute_quadrants = aggregate_metrics(records.groupby([PAIR_TYPE, QUADRANT])[NUM_CASES], axis="columns")
    with (out_dir / "mean_absolute_confusion_matrix_quadrants_over_runs.txt").open("w") as f:
        f.write("Mean absolute number of TP, FP, ... of each link type over all runs\n\n")
        f.write(tabulate(mean_absolute_quadrants, headers="keys"))
    mean_absolute_quadrants.to_pickle(out_dir / "mean_absolute_confusion_matrix_quadrants_over_runs.pkl")


def get_mention_context(dataset, idx, num_context_pre=2):
    doc_id, _ = idx
    sent_idx = dataset.mentions_action.at[idx, SENTENCE_IDX]

    # determine how many preceding and following sentences there are for the mention sentence in this document
    tokens = dataset.tokens.loc[doc_id, TOKEN]
    sent_idx_start = max(sent_idx - num_context_pre, 0)
    mention_context = tokens.loc[slice(sent_idx_start, sent_idx)].copy()

    # highlight the token span (or full sentence) of the mention
    mention = dataset.mentions_action.loc[idx]
    mention_context.loc[(sent_idx, mention[TOKEN_IDX_FROM])] = ">>>" + mention_context.loc[(sent_idx, mention[TOKEN_IDX_FROM])]
    mention_context.loc[(sent_idx, mention[TOKEN_IDX_TO] - 1)] = mention_context.loc[(sent_idx, mention[TOKEN_IDX_TO] - 1)] + "<<<"
    return " ".join(mention_context.values.tolist())


def get_document_context(dataset, idx, num_sentences=2):
    doc_id, _ = idx
    tokens = dataset.tokens.loc[doc_id, TOKEN]
    sent_idx_end = min(num_sentences, tokens.index.get_level_values(SENTENCE_IDX).max())
    document_context = " ".join(tokens.loc[slice(0, sent_idx_end)].values.tolist())
    return document_context


def _print_prediction_pairs(index_df: pd.DataFrame,
                           outcome_df: pd.DataFrame,
                           dataset: Dataset,
                           num_samples_per_quadrant: int,
                           serialization_dir: Path):
    out_dir = serialization_dir / "prediction_examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    outcomes = pd.concat([index_df, outcome_df], axis=1)

    # sample n instances from each quadrant
    outcomes.index.name = INSTANCE
    sampled_outcomes = outcomes.groupby([PAIR_TYPE, QUADRANT]).apply(lambda group: group.sample(min(len(group), num_samples_per_quadrant), random_state=0))
    sampled_outcomes = sampled_outcomes.reorder_levels([PAIR_TYPE, QUADRANT, INSTANCE]).sort_index().drop(columns=[PAIR_TYPE, QUADRANT])

    # convert the mention index columns into one again, because that's what the code below was written for TODO nasty
    sampled_outcomes["a-mention-idx"] = sampled_outcomes[[IDX_A_DOC, IDX_A_MENTION]].apply(lambda row: (row[IDX_A_DOC], int(row[IDX_A_MENTION])), axis=1)
    sampled_outcomes["b-mention-idx"] = sampled_outcomes[[IDX_B_DOC, IDX_B_MENTION]].apply(lambda row: (row[IDX_B_DOC], int(row[IDX_B_MENTION])), axis=1)
    sampled_outcomes.drop(columns=[IDX_A_DOC, IDX_A_MENTION, IDX_B_DOC, IDX_B_MENTION], inplace=True)

    # look up mention context and document context for each mention in each pair
    idx_a_info = sampled_outcomes["a-mention-idx"].apply(lambda v: pd.Series({IDX_A_DOC: get_document_context(dataset, v), IDX_A_MENTION: get_mention_context(dataset, v)}))
    idx_b_info = sampled_outcomes["b-mention-idx"].apply(lambda v: pd.Series({IDX_B_DOC: get_document_context(dataset, v), IDX_B_MENTION: get_mention_context(dataset, v)}))

    outcomes_with_text = pd.concat([sampled_outcomes, idx_a_info, idx_b_info], axis=1)
    outcomes_with_text.sort_index(inplace=True)
    outcomes_with_text.to_csv(out_dir / "prediction_examples.csv")

    # apply textwrap to columns to make it readable
    for col in [IDX_A_MENTION, IDX_B_MENTION, IDX_A_DOC, IDX_B_DOC]:
        outcomes_with_text[col] = outcomes_with_text[col].map(lambda s: textwrap.fill(s, width=30))

    for pair_type, df in outcomes_with_text.groupby(PAIR_TYPE):
        pair_type_dir = out_dir / f"{pair_type} pairs"
        pair_type_dir.mkdir(exist_ok=True)

        for quadrant, inner_df in df.groupby(QUADRANT):
            with (pair_type_dir / f"TXT_{quadrant}_prediction_examples.txt").open("w") as f:
                f.write(tabulate(inner_df, headers="keys", tablefmt="grid", showindex=False))
            with (pair_type_dir / f"TEX_{quadrant}_prediction_examples.tex").open("w") as f:
                f.write(tabulate(inner_df, headers="keys", tablefmt="latex", showindex=False))