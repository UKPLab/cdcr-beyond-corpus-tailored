from pathlib import Path

from python.handwritten_baseline.pipeline.data.base import BaselineDataLoaderStage


class FccLoaderBaseStage(BaselineDataLoaderStage):

    def __init__(self, pos, config, config_global, logger):
        super(FccLoaderBaseStage, self).__init__(pos, config, config_global, logger)

        self._sentence_level_data_dir = Path(config["sentence_level_data_dir"])
        assert self._sentence_level_data_dir.exists()