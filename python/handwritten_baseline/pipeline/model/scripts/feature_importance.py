from logging import Logger
from pathlib import Path
from typing import List, Any, Optional

import eli5
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from tabulate import tabulate

from python.handwritten_baseline.pipeline.model.classifier_clustering.pairwise_classifier_wrapper import \
    PredictOnTransformClassifierWrapper
from python.handwritten_baseline.pipeline.model.scripts.pipeline_instantiation import CLASSIFIER_PIPELINE_STEP_NAME


def get_feature_names_from_pipeline(p) -> List[str]:
    """
    Recursively gets feature names from an sklearn pipeline.

    There is no `get_feature_names` method for Pipeline objects, see https://stackoverflow.com/a/45602388 . As of May
    2020, this feature is discussed but never fully implemented since 4 years, see for example https://github.com/scikit-learn/enhancement_proposals/pull/18 .
    :param p:
    :return:
    """
    if isinstance(p, Pipeline):
        names = []
        for step_name, step in p.named_steps.items():
            names += get_feature_names_from_pipeline(step)
        return names
    elif isinstance(p, FeatureUnion):
        # FeatureUnions do have the `get_feature_names` method, but it fails if there is a pipeline inside the union
        # so we need to step inside the FeatureUnion.
        names = []
        for feature_name, feature in p.transformer_list:
            names += get_feature_names_from_pipeline(feature)
        return names
    elif hasattr(p, "get_feature_names"):
        # at this point we should have an actual feature object on our hands
        return p.get_feature_names()
    else:
        # we don't need any names from unrelated pipeline parts (classifiers, ...)
        return []


def get_named_component_of_pipeline(p: Any, q_name: str) -> Optional[Any]:
    """
    Recurses over a pipeline to find the first matching step or transformer in a feature with the given name.
    :param p: Pipeline or FeatureUnion object
    :param q_name: query name
    :return: the object with the given name, or None if it was not found
    """
    if isinstance(p, Pipeline):
        for step_name, step in p.named_steps.items():
            if step_name == q_name:
                return step
            obj = get_named_component_of_pipeline(step, q_name)
            if obj is not None:
                return obj
    elif isinstance(p, FeatureUnion):
        for feature_name, feature in p.transformer_list:
            if feature_name == q_name:
                return feature
            obj = get_named_component_of_pipeline(feature, q_name)
            if obj is not None:
                return obj
    return None


def analyze_feature_importance(pipelines: List[Pipeline],
                               serialization_dir: Path,
                               logger: Logger):
    feature_importances = []
    for i, p in enumerate(pipelines):
        # Analyze feature importance: This is quite a "scrapy" endeavour. We need to obtain the feature names in order (for
        # which sklearn provides no method) and we need to find the classifier step in the pipeline (again, no method).
        feature_names = get_feature_names_from_pipeline(p)
        classifier = get_named_component_of_pipeline(p, CLASSIFIER_PIPELINE_STEP_NAME)
        assert classifier is not None, "Pipeline broken? Could not find classifier."
        assert type(classifier) is PredictOnTransformClassifierWrapper, "Unexpected pipeline step type."
        try:
            feature_importance = eli5.explain_weights_df(classifier.classifier_, feature_names=feature_names)
            if feature_importance is None:
                raise ValueError
        except Exception as e:
            logger.warning(f"Could not determine feature importance for {repr(type(classifier.classifier_))}.", e)
            continue
        feature_importance["run"] = i
        feature_importances.append(feature_importance)

    if not feature_importances:
        logger.warning("No feature importances found.")
        return
    feature_importances = pd.concat(feature_importances)

    # write raw data to file
    all_importances_file = serialization_dir / "feature_importances.csv"
    feature_importances.to_csv(all_importances_file)

    # average weight by run and write it to file
    importances_aggregated = feature_importances.groupby("feature")["weight"].describe(percentiles=[])
    importances_aggregated.sort_values("mean", ascending=False, inplace=True)
    aggregated_importances_file = serialization_dir / "feature_importances_aggregated.txt"
    with aggregated_importances_file.open("w") as f:
        f.write(tabulate(importances_aggregated, headers="keys"))