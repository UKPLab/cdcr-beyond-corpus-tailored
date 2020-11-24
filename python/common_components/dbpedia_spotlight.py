from python.pipeline import ComponentBase


class DbPediaSpotlight(ComponentBase):

    def __init__(self, config, config_global, logger):
        super(DbPediaSpotlight, self).__init__(config, config_global, logger)

        self._endpoint = config["endpoint"] # type: str
        self._wait_between_requests_seconds = float(config.get("wait_between_requests_seconds", 0))
        self._confidence = config.get("confidence", 0.7)  # type: float
        self._support = config.get("support", 0)  # type: int

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def wait_between_requests_seconds(self) -> float:
        return self._wait_between_requests_seconds

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def support(self) -> int:
        return self._support
