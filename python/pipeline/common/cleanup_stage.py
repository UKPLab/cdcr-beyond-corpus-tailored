from typing import Dict, Tuple, Any

from python.pipeline import ComponentBase
from python.pipeline.pipeline import PipelineStage


class CleanupStage(PipelineStage):
    """
    Calls cleanup() on all live objects in the pipeline.
    """

    def __index__(self, pos, config, config_global, logger):
        super(CleanupStage, self).__init__(pos, config, config_global, logger)

    def files_produced(self) -> Dict[str, Tuple[str, str]]:
        return {}

    def run(self, live_objects: Dict[str, Any]):
        for component in live_objects.values():
            if isinstance(component, ComponentBase):
                component.clean_up()


component = CleanupStage
