from pathlib import Path
from typing import Optional, Union, List, Tuple, Any, Dict

import pandas as pd
from sklearn.metrics import log_loss, precision_recall_fscore_support
from tabulate import tabulate

from python.handwritten_baseline.pipeline.model.scripts.evaluation_utils import run_conll_evaluation
from python.handwritten_baseline import PREDICTION, LABEL, INSTANCE, IDX_A_DOC, IDX_A_MENTION, IDX_B_DOC, IDX_B_MENTION
from python.handwritten_baseline.pipeline.model.data_prep.mention_pair_generator import MentionPairGenerator
from python.handwritten_baseline.pipeline.model.data_prep.pipeline_data_input import \
    convert_X_and_y_to_internal_pipeline_input_fmt, PAIR_PREDICTION_RANDOM_SEED, create_gold_clustering
from python.util.util import get_date_based_subdirectory


def obtain_final_coref_metrics(gold_clusters: pd.Series,
                               system_clusters: pd.Series,
                               data_split: str,
                               coref_metrics: Optional[Union[str, List[str]]],
                               serialization_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Runs the CoNLL evaluation script for CDCR, once with metadoc True and once False.
    :param gold_clusters:
    :param system_clusters:
    :param data_split:
    :param coref_metrics:
    :param serialization_dir: if given, CoNLL files and a metric overview will be written to this directory
    :return: the metrics in a dataframe
    """
    all_metrics = []
    for meta_doc in [True, False]:
        meta_doc_descr = "cross_doc" if meta_doc else "within_doc"

        if serialization_dir is not None:
            sub_serialization_dir = serialization_dir / meta_doc_descr
            sub_serialization_dir.mkdir(exist_ok=True)
        else:
            sub_serialization_dir = None

        metrics = run_conll_evaluation(gold_clusters,
                                       system_clusters,
                                       single_meta_document=meta_doc,
                                       metrics=coref_metrics,
                                       output_dir=sub_serialization_dir)
        metrics = metrics.unstack(level="measure")

        # write metric overview to file if desired
        if sub_serialization_dir is not None:
            filename = "_".join([data_split, meta_doc_descr]) + ".txt"
            with (sub_serialization_dir / filename).open("w") as f:
                f.write(tabulate(metrics, headers="keys"))

        metrics["meta-doc"] = meta_doc
        all_metrics.append(metrics)
    all_metrics = pd.concat(all_metrics).reset_index()
    return all_metrics


class CrossDocCorefScoring:

    def __init__(self,
                 metrics: Optional[Union[str, List[str]]] = "all",
                 only_lea_f1_for_cv: bool = False,
                 serialization_dir: Optional[Path] = None):
        """

        :param metrics: family of metrics to report ("muc", "ceafe", "lea", ...)
        :param only_lea_f1_for_cv: If False, returns metrics as a pd.Series. If True, returns only LEA F1. This is useful for
                            cross-validation. If True, this parameter overrides the `metrics` parameter.
        :param serialization_dir: Destination to write CoNLL files and metric overviews to.
        """
        self.only_lea_f1_for_cv = only_lea_f1_for_cv
        if self.only_lea_f1_for_cv:
            self.metrics = "lea"
        else:
            self.metrics = metrics

        self.serialization_dir = serialization_dir

    def __call__(self, *args, **kwargs) -> Union[float, Tuple[pd.DataFrame, Any]]:
        assert len(args) == 3
        estimator, X, y = args

        all_mentions_to_gold_events = pd.concat(y).sort_index()
        gold_clusters = create_gold_clustering(all_mentions_to_gold_events)

        system_clusters = estimator.predict(X)

        # Our clustering approach cannot produce multiple clusters for one mention. We still need to map the integer
        # cluster identifiers in lists.
        gold_clusters = gold_clusters.map(lambda v: [v])
        system_clusters = system_clusters.map(lambda v: [v])

        if self.only_lea_f1_for_cv:
            metrics = run_conll_evaluation(gold_clusters, system_clusters, single_meta_document=True, metrics="lea")
            return metrics.loc["lea", "f1"]
        else:
            metrics = obtain_final_coref_metrics(gold_clusters,
                                                 system_clusters,
                                                 "eval",
                                                 coref_metrics=self.metrics,
                                                 serialization_dir=self.serialization_dir)
            return metrics, None    # no extras here


class MentionPairScoring:

    def __init__(self,
                 mpg_prediction_config: Optional[Dict],
                 return_neg_log_loss_for_cv: bool = False,
                 serialization_dir: Optional[Path] = None):
        self.mpg_prediction_config = {} if mpg_prediction_config is None else mpg_prediction_config

        # do a quick instantiation check here to validate parameters - fail early
        _ = MentionPairGenerator(**self.mpg_prediction_config)

        self.return_neg_log_loss_for_cv = return_neg_log_loss_for_cv
        self.serialization_dir = serialization_dir

    def __call__(self, *args, **kwargs) -> Union[float, Tuple[pd.DataFrame, Any]]:
        assert len(args) == 3
        estimator, X, y = args

        y_pred = estimator.predict(X)

        generator = MentionPairGenerator(**self.mpg_prediction_config,
                                         serialization_dir=get_date_based_subdirectory(self.serialization_dir))
        _, pairs, y_true, _ = convert_X_and_y_to_internal_pipeline_input_fmt(generator=generator,
                                                                             X=X,
                                                                             y=y,
                                                                             random_state=PAIR_PREDICTION_RANDOM_SEED)
        assert len(y_pred) == len(y_true)

        if self.return_neg_log_loss_for_cv:
            return -log_loss(y_true, y_pred, labels=[False, True])
        else:
            # compute metrics
            y_pred_binarized = (y_pred >= 0.5).reshape((-1, 1))
            p, r, f1, support = precision_recall_fscore_support(y_true, y_pred_binarized, pos_label=True, average="binary", zero_division=0)
            metrics = pd.DataFrame([{"metric": "pairs", "precision": p, "recall": r, "f1": f1, "support": support}])

            # prepare predictions for further analysis
            outcomes = pd.DataFrame(pairs, columns=pd.Index([IDX_A_DOC, IDX_A_MENTION, IDX_B_DOC, IDX_B_MENTION]))
            outcomes[PREDICTION] = y_pred_binarized
            outcomes[LABEL] = y_true
            outcomes.index.rename(INSTANCE, inplace=True)
            outcomes.to_pickle(str(self.serialization_dir / "outcomes.pkl"))

            return metrics, outcomes