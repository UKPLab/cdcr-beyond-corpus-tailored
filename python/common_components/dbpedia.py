import time
from typing import Dict
from urllib.error import HTTPError

from SPARQLWrapper import SPARQLWrapper, JSON
from SPARQLWrapper.SPARQLExceptions import QueryBadFormed, Unauthorized, EndPointNotFound, URITooLong, \
    EndPointInternalError

from python.pipeline import ComponentBase


class DbPedia(ComponentBase):

    def __init__(self, config, config_global, logger):
        super(DbPedia, self).__init__(config, config_global, logger)

        self._endpoint = config["sparql_endpoint"]
        self._graph_iri = config["graph_iri"]
        self._wait_between_requests_seconds = float(config.get("wait_between_requests_seconds", 0))

        self._cache = self._provide_cache("dbpedia", bind_parameters=[self.endpoint, self.graph_iri])
        self._time_of_last_query = 0

    def query(self, query: str) -> Dict:
        """
        Performs cached SPARQL query to DBpedia.
        """
        if query not in self._cache:
            sparql = SPARQLWrapper(self.endpoint, returnFormat=JSON)
            sparql.setQuery(query)

            # apply rate limiting: make sure at least self._wait_between_requests_seconds seconds are between each request
            now = time.time()
            time_to_sleep = max(0, self._wait_between_requests_seconds - (now - self._time_of_last_query))
            time.sleep(time_to_sleep)

            try:
                results = sparql.query().convert()
                self._cache[query] = results
            except (QueryBadFormed, Unauthorized, EndPointNotFound, URITooLong, EndPointInternalError, HTTPError) as e:
                self.logger.error(e)
                raise ValueError from e
            finally:
                self._time_of_last_query = now
        else:
            results = self._cache[query]
        return results

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def graph_iri(self) -> str:
        return self._graph_iri