from typing import Dict, Tuple, Any

from python.pipeline.pipeline import PipelineStage


class DummyFeederStage(PipelineStage):
    """
    Any configuration key-value pairs passed to this stage in `files_produced` will be returned as files produced by
    this stage. Configuration key-value pairs passed in `live_objects` will be put in live objects.
    """

    def __index__(self, pos, config, config_global, logger):
        super(DummyFeederStage, self).__init__(pos, config, config_global, logger)

    def files_produced(self) -> Dict[str, Tuple[str, str]]:
        return {k: v for k, v in self.config.get("files_produced", {}).items()}

    def run(self, live_objects: Dict[str, Any]):
        live_objects.update(self.config.get("live_objects", {}))

component = DummyFeederStage
