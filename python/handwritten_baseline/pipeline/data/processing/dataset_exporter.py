import pickle
from typing import Dict

from python import TOKEN, DOCUMENT_ID, EVENT, SUBTOPIC, TOPIC_ID, COLLECTION
from python.handwritten_baseline.pipeline.data.base import Dataset, BaselineDataProcessorStage
from python.pipeline import ID, TIMESTAMP


class DatasetExporterStage(BaselineDataProcessorStage):
    """
    Writes a dataset to disk using pickle.
    """

    def __init__(self, pos, config, config_global, logger):
        super(DatasetExporterStage, self).__init__(pos, config, config_global, logger)

        dataset_name = config["dataset_name"]
        split_name = config["split"]
        filename = "_".join([dataset_name, split_name, "preprocessed", config_global[ID], config_global[TIMESTAMP]]) + ".pickle"
        self._output_file = self.stage_disk_location / filename

        self._gold_doc_clustering_file = self.stage_disk_location / ("_".join([dataset_name, split_name, config_global[ID], config_global[TIMESTAMP], "gold_transitive_closure_doc_clustering"]) + ".pkl")

        self._do_export_plaintext_documents = config.get("export_plaintext_documents_as_well", False)
        if self._do_export_plaintext_documents:
            self._plaintext_dir = self.stage_disk_location / "plaintext_documents"
            self._plaintext_dir.mkdir(parents=True)

    def _export_dataset(self, dataset: Dataset):
        """
        Export preprocessed dataset and gold clustering.
        """
        with self._output_file.open("wb") as f:
            pickle.dump(dataset, f)


        # put the topic/subtopic information into the document identifier
        documents = dataset.documents
        documents.index = documents.index.droplevel(DOCUMENT_ID)
        mentions_action = dataset.mentions_action.reset_index().merge(documents.reset_index(), left_on=DOCUMENT_ID, right_on=DOCUMENT_ID)

        subtopic_column = SUBTOPIC
        if COLLECTION in documents.columns:
            self.logger.info("Found 'collection' column in documents dataframe (FCC corpus?). Dataset will be exported using 'collection' as a replacement for subtopics.")
            subtopic_column = COLLECTION

        get_conflated_doc_id = lambda df: df[TOPIC_ID].astype(str).str.replace("_", "-") + "_" + df[subtopic_column].astype(str) + "_" + df[DOCUMENT_ID].astype(str)
        mentions_action[DOCUMENT_ID] = get_conflated_doc_id(mentions_action)

        # determine transitive closure event coreference relation w.r.t. documents to create *the* gold preclustering
        mentions_and_documents = mentions_action[[DOCUMENT_ID, EVENT]].drop_duplicates()

        gold_doc_clustering = []
        for event, docs in mentions_and_documents.groupby(EVENT):
            docs_for_event = set(docs[DOCUMENT_ID].values)

            indices_of_intersecting_clusters = []
            for idx, doc_cluster in enumerate(gold_doc_clustering):
                if len(doc_cluster & docs_for_event) > 0:
                    indices_of_intersecting_clusters.append(idx)

            # get rid of old clusters
            new_cluster = set()
            for k, idx in enumerate(indices_of_intersecting_clusters):
                doc_cluster = gold_doc_clustering.pop(idx - k)
                new_cluster |= doc_cluster

            new_cluster |= docs_for_event
            gold_doc_clustering.append(new_cluster)

        # convert the partitioning in the usual list-of-lists format
        gold_doc_clustering = [list(doc_cluster) for doc_cluster in gold_doc_clustering]
        with open(self._gold_doc_clustering_file, "wb") as f:
            pickle.dump(gold_doc_clustering, f)

    def _export_plaintext_documents(self, dataset):
        for doc_id, tokens in dataset.tokens.groupby(DOCUMENT_ID)[TOKEN]:
            doc_text = " ".join(tokens.values)
            with (self._plaintext_dir / f"{doc_id}.txt").open("w") as f:
                f.write(doc_text)

    def _process_dataset(self,
                         dataset: Dataset,
                         live_objects: Dict) -> Dataset:
        # final assertion before exporting: make sure that all documents contain at least one action mention (this
        # broke our neck several times)
        documents_set = set(dataset.documents[DOCUMENT_ID].unique())
        mentions_set = set(dataset.mentions_action.index.get_level_values(DOCUMENT_ID).unique())
        assert len(documents_set) == len(documents_set & mentions_set)

        self._export_dataset(dataset)

        if self._do_export_plaintext_documents:
            self._export_plaintext_documents(dataset)

        return dataset


component = DatasetExporterStage
