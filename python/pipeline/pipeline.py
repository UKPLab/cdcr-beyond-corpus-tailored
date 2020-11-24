from pathlib import Path
from typing import Dict, Any

from python.pipeline import ComponentBase


class PipelineStage(ComponentBase):
    def __init__(self, pos, config, config_global, logger):
        """
        Pipeline stage constructor
        :param pos: position in the pipeline
        :param config:
        :param config_global:
        :param logger:
        """
        super(PipelineStage, self).__init__(config, config_global, logger)

        self._pos = pos
        self.config_working_dir = config_global["config_working_dir"]
        self.stage_disk_location = Path(self._provide_disk_location(f"{self.position}_{self.short_name}", make_dir=True))

    @property
    def position(self):
        return self._pos

    def requires_files(self, provided: Dict[str, Path]):
        pass

    def files_produced(self) -> Dict[str, Path]:
        """
        Returns files/directories produced by this pipeline stage, identified by a name.
        :return: for example: {"input-data": "/foo/bar"}
        """
        # TODO confirm after running the pipeline that these files were actually written
        return {}

    def run(self, live_objects: Dict[str, Any]):
        """
        Run pipeline stage.
        :param live_objects: mutable dict of objects output from stages further up the pipeline
        :return:
        """
        pass
