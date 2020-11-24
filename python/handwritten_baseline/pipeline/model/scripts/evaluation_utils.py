import shutil
import tempfile
from typing import Union

import python.util.conll_scorer.scorer as conll_scorer
from python.handwritten_baseline.pipeline.data.loader.ecb_reader_utils import *
from python.handwritten_baseline.pipeline.data.loader.football_reader_utils import *


def export_conll(f, clustering: pd.Series, single_meta_document: bool = False):
    """
    Exports a clustering result into CoNLL format.
    :param f: destination file handle
    :param clustering: clustering output
    :param single_meta_document: Throw away {document -> sentence} mapping and put all sentences in one meta document. Necessary for cross-doc evaluation.
    :return:
    """
    # the first index level must be the document id
    assert clustering.index.names[0] == DOCUMENT_ID

    clustering_w_parens = clustering.map(lambda cl_ids: "(" + ",".join(str(i) for i in cl_ids) + ")", na_action="ignore")

    # remove doc-id from index, set to same value everywhere, then add it back to where it was
    if single_meta_document:
        # create a new index which is a concatenation of all multi-index columns, separated by an underscore
        index_cols_joined = clustering_w_parens.index.to_frame().astype(str).apply(lambda row: "_".join(row), axis=1)
        joined_index = pd.Index(index_cols_joined.values, name=MENTION_ID)

        # apply it
        clustering_w_parens.index = joined_index

        # to conform to the expected dataframe format, add a multi-index column DOCUMENT_ID which says "metadoc" for each row
        clustering_w_parens = pd.concat([clustering_w_parens], keys=["metadoc"], names=[DOCUMENT_ID])

    for doc_id, group in clustering_w_parens.groupby(level=DOCUMENT_ID):
        f.write("#begin document(" + doc_id + ");\n")

        # merge multiindex levels into one
        group.index = group.index.map(lambda row: "_".join([str(v) for v in row]))
        csv = group.to_csv(sep="\t", header=False, index=True, na_rep="-")
        f.write(csv)

        f.write("#end document;\n")


def run_conll_evaluation(gold_clustering: pd.Series,
                         system_prediction: pd.Series,
                         single_meta_document: bool = True,
                         metrics: Optional[Union[str, List[str]]] = "all",
                         output_dir: Optional[Path] = None) -> pd.Series:
    """
    Runs the standard CoNLL evaluation scripts. Returns per-metric results in a pandas Series.
    :param gold_clustering:
    :param system_prediction:
    :param single_meta_document:
    :param metrics:
    :param output_dir: Destination for CoNLL files. If None, a temporary location is used.
    :return:
    """

    o_dir = output_dir if output_dir is not None else Path(tempfile.mkdtemp(prefix="conll_"))

    with (o_dir / "gold.conll").open("w") as f_gold, \
            (o_dir / "system.conll").open("w") as f_system:
        export_conll(f_gold, gold_clustering, single_meta_document=single_meta_document)
        export_conll(f_system, system_prediction, single_meta_document=single_meta_document)

        # temp files are buffered for efficiency reasons, need to flush to truly write changes to disk
        f_gold.flush()
        f_system.flush()

        # run evaluation
        result = conll_scorer.evaluate(f_gold.name,
                                       f_system.name,
                                       metrics,
                                       NP_only=False,
                                       remove_nested=False,
                                       keep_singletons=True)

    # delete temporary location
    if output_dir is None:
        shutil.rmtree(o_dir)

    return result