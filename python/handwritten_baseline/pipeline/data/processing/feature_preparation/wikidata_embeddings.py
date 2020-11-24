from pathlib import Path
from pprint import pformat
from typing import Dict

import ijson
import numpy as np
from tqdm import tqdm

from python.handwritten_baseline import WIKIDATA_QID, WIKIDATA_EMBEDDINGS
from python.handwritten_baseline.pipeline.data.base import Dataset, BaselineDataProcessorStage


class WikidataEmbeddingFeaturePreparationStage(BaselineDataProcessorStage):
    """
    For the python.handwritten_baseline.features.wikidata_embedding_distance.WikidataEmbeddingDistanceFeature, we need
    Wikidata embeddings for each linked entity. It's more efficient and more convenient for configuring training if we
    look up the embeddings at data preparation time rather than when computing the feature.
    """

    def __init__(self, pos, config, config_global, logger):
        super(WikidataEmbeddingFeaturePreparationStage, self).__init__(pos, config, config_global, logger)

        self._json_index_file = Path(config["json_index"])
        self._embedding_matrix_file = Path(config["embedding_npy"])

        assert self._json_index_file.exists() and self._json_index_file.is_file()
        assert self._embedding_matrix_file.exists() and self._embedding_matrix_file.is_file()

    def _process_dataset(self,
                         dataset: Dataset,
                         live_objects: Dict) -> Dataset:
        WIKIDATA_NAMESPACE = "http://www.wikidata.org/entity/"

        # determine for which QIDs we need to look up embeddings
        set_of_wikidata_qids = set()
        for df in [dataset.mentions_action,
                   dataset.mentions_time,
                   dataset.mentions_location,
                   dataset.mentions_participants,
                   dataset.mentions_other]:
            if df is None:
                continue
            assert WIKIDATA_QID in df.columns, "Need to entity link against Wikidata first!"
            set_of_wikidata_qids |= set(df[WIKIDATA_QID].loc[df[WIKIDATA_QID].notna()].unique())

        wikidata_iris = {f"<{WIKIDATA_NAMESPACE}{qid}>": qid for qid in set_of_wikidata_qids}

        # load the relevant embedding vectors: use mmap_mode="r" to not load gigabytes of stuff into RAM
        mat_embedding = np.load(self._embedding_matrix_file, mmap_mode="r")
        num_terms = mat_embedding.shape[0]

        # Check the JSON index to find the indices of these QIDs in the pretrained embedding matrix. Use ijson to parse
        # the file incrementally, avoiding to load 3GB of JSON into RAM.
        qid_to_mat_embedding_index = {}
        qid_to_mat_embedding_subset_index = {}
        with self._json_index_file.open("rb") as f:
            for i, term in tqdm(enumerate(ijson.items(f, "item")), desc="Looking up QIDs in embedding index",
                                mininterval=10, total=num_terms, unit="terms"):
                try:
                    unicode_term = term.encode().decode("unicode_escape").strip()
                except UnicodeDecodeError as e:
                    self.logger.warn(e)
                    continue
                if unicode_term in wikidata_iris.keys():
                    qid = wikidata_iris.pop(unicode_term)
                    qid_to_mat_embedding_index[qid] = i
                    qid_to_mat_embedding_subset_index[qid] = len(qid_to_mat_embedding_subset_index)

                    # bail early if done
                    if not wikidata_iris:
                        self.logger.info("All QIDs found!")
                        break
        if wikidata_iris:
            self.logger.warning(
                f"The following {len(wikidata_iris)} Wikidata entities were not found in the pretrained embedding index:\n" + pformat(
                    wikidata_iris))

        # look up relevant embeddings
        mat_embedding_subset = mat_embedding[list(qid_to_mat_embedding_index.values())]

        # and we're done
        wikidata_embeddings = (qid_to_mat_embedding_subset_index, mat_embedding_subset)
        dataset.set(WIKIDATA_EMBEDDINGS, wikidata_embeddings)

        return dataset


component = WikidataEmbeddingFeaturePreparationStage
