from pathlib import Path

import pandas as pd
from overrides import overrides

from python import EVENT_ID, DOCUMENT_ID, SUBTOPIC, EVENT
from python.handwritten_baseline.pipeline.data.loader import gvc_reader_utils
from python.handwritten_baseline.pipeline.data.base import BaselineDataLoaderStage, Dataset


class GvcLoaderStage(BaselineDataLoaderStage):
    root_data_path = Path("resources") / "data" / "gun_violence"

    def __init__(self, pos, config, config_global, logger):
        super(GvcLoaderStage, self).__init__(pos, config, config_global, logger)

        self._gvc_root_dir = Path(config["gvc_root_dir"])
        assert self._gvc_root_dir.exists()

        self._gvc_split_csv = self._gvc_root_dir / config["gvc_split_csv_filename"]
        assert self._gvc_split_csv.exists()

        self._drop_0_cluster = config["drop_0_cluster"]

    @overrides
    def _load_dataset(self) -> Dataset:
        # load full dataset
        documents, contents, mentions = gvc_reader_utils.load_gvc_dataset(self._gvc_root_dir / "GVC_gold.conll",
                                                                          doc_to_subtopic_file=self._gvc_root_dir / "gvc_doc_to_event.csv")

        # look up the events for this split and which documents belong to which event, then combine the two into the
        # documents which are part of this split
        split = pd.read_csv(self._gvc_split_csv, index_col=0, header=None, names=[EVENT_ID])
        docs_of_split = documents.loc[documents.index.get_level_values(SUBTOPIC).isin(split[EVENT_ID].astype(str))].set_index(DOCUMENT_ID)

        # return only instances of this split
        documents = documents.loc[documents[DOCUMENT_ID].isin(docs_of_split.index)].sort_index()
        contents = contents.loc[docs_of_split.index].sort_index()
        mentions_action = mentions.loc[docs_of_split.index].sort_index()

        if self._drop_0_cluster:
            mentions_action = mentions_action.loc[mentions_action[EVENT] != 0]

        dataset = Dataset(documents,
                          contents,
                          mentions_action)
        return dataset

    # exporting incidents and documents for a split:
    # df = dataset.documents.index.to_frame().reset_index(drop=True)
    # df.drop(columns="topic-id", inplace=True)
    # df = df.sort_values(["subtopic", "doc-id"])
    # df["subtopic"] = df["subtopic"].astype(str)
    # df.loc[((df["subtopic"].duplicated(keep=False)) & df["subtopic"].duplicated()), "subtopic"] = ""
    # f = open("split.txt", "w")
    # f.write(df, showindex=False, tablefmt="latex"))
    # f.close()


component = GvcLoaderStage
