import pandas as pd
from overrides import overrides

from python import *
from python.handwritten_baseline.pipeline.data.base import BaselineDataLoaderStage, Dataset


class HyperlinksLoaderStage(BaselineDataLoaderStage):

    def __init__(self, pos, config, config_global, logger):
        super(HyperlinksLoaderStage, self).__init__(pos, config, config_global, logger)

        self.page_infos_file = config["page_infos"]
        self.tokens_file = config["tokens"]
        self.hyperlinks_file = config["hyperlinks"]

    @overrides
    def _load_dataset(self) -> Dataset:
        page_infos = pd.read_parquet(self.page_infos_file)  # type: pd.DataFrame
        hyperlinks = pd.read_parquet(self.hyperlinks_file)  # type: pd.DataFrame
        tokens = pd.read_parquet(self.tokens_file)[TOKEN]  # type: pd.Series

        # prepare documents df
        documents = page_infos.reset_index()[["url-normalized", PUBLISH_DATE]].rename(columns={"url-normalized": DOCUMENT_ID})
        documents[TOPIC_ID] = "hyperlinks"
        documents[SUBTOPIC] = "subt0"
        documents.set_index([TOPIC_ID, SUBTOPIC, DOCUMENT_ID], inplace=True)
        documents.sort_index(inplace=True)
        documents[DOCUMENT_ID] = documents.index.get_level_values(DOCUMENT_ID)  # add doc-id back as a data column

        tokens = tokens.to_frame(TOKEN)     # type: pd.DataFrame
        tokens.index.names = [DOCUMENT_ID, SENTENCE_IDX, TOKEN_IDX]
        tokens.sort_index(inplace=True)

        hyperlinks_by_doc = hyperlinks.groupby("url-normalized").apply(lambda v: v.reset_index(drop=True))
        hyperlinks_by_doc.index.names = [DOCUMENT_ID, MENTION_ID]
        hyperlinks_by_doc.sort_index(inplace=True)
        hyperlinks_by_doc.rename(columns={"to-url-normalized": EVENT}, inplace=True)

        dataset = Dataset(documents,
                          tokens,
                          hyperlinks_by_doc)
        return dataset


component = HyperlinksLoaderStage
