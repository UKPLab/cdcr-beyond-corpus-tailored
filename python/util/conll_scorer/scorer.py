from typing import List, Union, Optional

import pandas as pd

from python.util.conll_scorer.conll import reader
from python.util.conll_scorer.eval import evaluator


def evaluate(key_file, sys_file, metrics: Optional[Union[str, List[str]]], NP_only=False, remove_nested=False, keep_singletons=True):
    available_metrics = {'lea': evaluator.lea, 'muc': evaluator.muc, 'bcub': evaluator.b_cubed, 'ceafe': evaluator.ceafe}
    if metrics is None or metrics == "all":
        metrics_used = available_metrics
    else:
        metrics_used = {k: v for k, v in available_metrics.items() if k in metrics}

    doc_coref_infos = reader.get_coref_infos(key_file, sys_file, NP_only, remove_nested, keep_singletons)

    index = pd.MultiIndex.from_product([list(metrics_used.keys()), ["precision", "recall", "f1"]], names=["metric", "measure"])
    evaluation = pd.Series(index=index)

    conll = 0
    conll_subparts_num = 0
    for name, metric in metrics_used.items():
        recall, precision, f1 = evaluator.evaluate_documents(doc_coref_infos, metric, beta=1)
        if name in {"muc", "bcub", "ceafe"}:
            conll += f1
            conll_subparts_num += 1

        evaluation[(name, "precision")] = precision
        evaluation[(name, "recall")] = recall
        evaluation[(name, "f1")] = f1

    if conll_subparts_num == 3:
        conll /= 3
        evaluation[("conll", "f1")] = conll

    return evaluation
