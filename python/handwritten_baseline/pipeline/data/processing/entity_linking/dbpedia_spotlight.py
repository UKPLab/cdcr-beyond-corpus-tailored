from typing import Optional, Dict

import pandas as pd
import spotlight

from python import CHARS_START, CHARS_END
from python.common_components import DBPEDIA_SPOTLIGHT, DBPEDIA
from python.common_components.dbpedia import DbPedia
from python.common_components.dbpedia_spotlight import DbPediaSpotlight
from python.handwritten_baseline import DBPEDIA_URI, MENTION_TEXT
from python.handwritten_baseline import WIKIDATA_QID
from python.handwritten_baseline.pipeline.data.processing.entity_linking.base import BaseEntityLinkingStage


class DbPediaSpotlightLinkingStage(BaseEntityLinkingStage):
    """
    Runs DBPedia Spotlight on each document. Mention spans found by DBPedia are mapped to those determined in previous
    data processing steps as best as possible. Additionally runs SPARQL queries against DBPedia to find the
    corresponding Wikidata QID for each entity.
    """

    def __init__(self, pos, config, config_global, logger):
        super(DbPediaSpotlightLinkingStage, self).__init__(pos, config, config_global, logger, "dbpedia_spotlight")

    def _query_entity_linker(self, doc_detokenized: str, live_objects: Dict) -> Optional[object]:
        dbpedia_spotlight = live_objects[DBPEDIA_SPOTLIGHT]  # type: DbPediaSpotlight
        try:
            return spotlight.annotate(dbpedia_spotlight.endpoint,
                                      doc_detokenized,
                                      confidence=dbpedia_spotlight.confidence,
                                      support=dbpedia_spotlight.support)
        except spotlight.SpotlightException as e:
            if "No Resources found in spotlight response" in repr(e):
                return None
            else:
                raise ValueError from e

    def _look_up_sameas(self, resource: str, dbpedia: DbPedia) -> str:
        """

        :param resource: DBPedia resource URI
        :return: wikidata QID
        """
        query = f"""
PREFIX owl: <http://www.w3.org/2002/07/owl#>

SELECT *
FROM <{dbpedia.graph_iri}>
WHERE {{
    {{<{resource}> owl:sameAs ?o}} UNION {{?o owl:sameAs <{resource}>}}
    FILTER(STRSTARTS(STR(?o), "http://wikidata.dbpedia.org/resource/"))
}}
"""
        results = dbpedia.query(query)

        qid = None
        data = results["results"]["bindings"]
        if data:
            row = data[0]
            qid = row["o"]["value"].split("/")[-1]
        if qid is None:
            self.logger.warning(f"No Wikidata equivalent found for {resource}")
        return qid

    def _get_waiting_time_between_requests_seconds(self, live_objects: Dict) -> float:
        dbpedia_spotlight =  live_objects[DBPEDIA_SPOTLIGHT]    # type: DbPediaSpotlight
        return dbpedia_spotlight.wait_between_requests_seconds

    def _convert_el_response_to_dataframe(self, obj, live_objects: Dict) -> pd.DataFrame:
        dbpedia = live_objects[DBPEDIA] # type: DbPedia

        # Source of these facts: http://succeed-project.eu/wiki/index.php/DBPedia_Spotlight
        #   Support: expresses how prominent this entity is. Based on the number of inlinks in Wikipedia.
        #   percentageOfSecondRank: measure how much the winning entity has won by taking contextualScore_2ndRank / contextualScore_1stRank, which means the lower this score, the further the first ranked entity was "in the lead"
        response_df = pd.DataFrame(obj)
        response_df.rename(columns={"offset": CHARS_START, "URI": DBPEDIA_URI, "surfaceForm": MENTION_TEXT},
                           inplace=True)
        response_df[CHARS_END] = response_df[CHARS_START] + response_df[MENTION_TEXT].str.len()

        # for each entity, look up the wikidata QID in DBPedia
        response_df[WIKIDATA_QID] = response_df[DBPEDIA_URI].apply(lambda uri: self._look_up_sameas(uri, dbpedia))

        return response_df


component = DbPediaSpotlightLinkingStage