from collections import OrderedDict
from typing import Dict

import pandas as pd
from toposort import toposort_flatten, CircularDependencyError
from tqdm import tqdm

from python.common_components import DBPEDIA
from python.common_components.dbpedia import DbPedia
from python.handwritten_baseline import DBPEDIA_URI, LATITUDE, LONGITUDE, GEO_HIERARCHY
from python.handwritten_baseline.pipeline.data.base import Dataset, BaselineDataProcessorStage


class LocationEnhanceStage(BaselineDataProcessorStage):
    """
    Enhances location mentions by adding latitude/longitude from DBPedia.
    """

    def __init__(self, pos, config, config_global, logger):
        super(LocationEnhanceStage, self).__init__(pos, config, config_global, logger)

    def _look_up_coordinates(self, resource: str, dbpedia: DbPedia) -> pd.Series:
        """

        :param resource: DBPedia resource URI
        :return: series containing the location
        """
        # DBPedia doesn't have the same geo properties for each entry, so we need to get them all, optionally, then
        # pick from what's there
        query = f"""
PREFIX geo: <http://www.georss.org/georss/>
PREFIX dbp: <http://dbpedia.org/property/>
PREFIX w3g: <http://www.w3.org/2003/01/geo/wgs84_pos#>

SELECT *
FROM <{dbpedia.graph_iri}>
WHERE {{
    OPTIONAL {{ <{resource}> geo:point ?p . }}
    OPTIONAL {{ <{resource}> dbp:latitude ?dbpLatitude . }}
    OPTIONAL {{ <{resource}> dbp:longitude ?dbpLongitude . }}
    OPTIONAL {{ <{resource}> dbp:latDeg ?latDeg . }}
    OPTIONAL {{ <{resource}> dbp:latMin ?latMin . }}
    OPTIONAL {{ <{resource}> dbp:lonDeg ?lonDeg . }}
    OPTIONAL {{ <{resource}> dbp:lonMin ?lonMin . }}
    OPTIONAL {{ <{resource}> w3g:lat ?w3gLatitude . }}
    OPTIONAL {{ <{resource}> w3g:long ?w3gLongitude . }}
}}
"""
        results = dbpedia.query(query)
        lat, lon = None, None

        data = results["results"]["bindings"]
        if data:
            row = data[0]
            if "p" in row:
                # coordinate_str is of format "34.1 -118.33333"
                value = row["p"]["value"]
                lat, lon = [float(f) for f in value.split()]
            elif "dbpLatitude" in row and "dbpLongitude" in row:
                lat = float(row["dbpLatitude"]["value"])
                lon = float(row["dbpLongitude"]["value"])
            elif "w3gLatitude" in row and "w3gLongitude" in row:
                lat = float(row["w3gLatitude"]["value"])
                lon = float(row["w3gLongitude"]["value"])
            elif "latDeg" in row and "lonDeg" in row and not ("latMin" in row and "lonMin" in row):
                # for some reason, this case exists
                lat = float(row["latDeg"]["value"])
                lon = float(row["lonDeg"]["value"])
            elif all(p in row for p in ["latDeg", "latMin", "lonDeg", "lonMin"]):
                lat = float(row["latDeg"]["value"]) + float(row["latMin"]["value"]) / 60
                lon = float(row["lonDeg"]["value"]) + float(row["lonMin"]["value"]) / 60
        if lat is None or lon is None:
            self.logger.warning(f"No location found for {resource}")
        elif lat < -90 or lat > 90 or lon < -180 or lon > 180:
            self.logger.warning(f"Illegal coordinates for {resource}: {lat}, {lon}")
            lat, lon = None, None

        ser = pd.Series({LATITUDE: lat, LONGITUDE: lon})
        return ser

    def _look_up_geographic_hierarchy(self, resource: str, dbpedia: DbPedia) -> pd.Series:
        """
        Returns geographic hierarchy for a resource, i.e. [dbo:San Francisco, dbo:California, dbo:United States]
        :param resource: DBpedia resource
        :return: hierarchy
        """
        # recursively get isPartOf relation from DBpedia
        dependencies = OrderedDict()
        queue = [resource]
        visited = set()
        while queue:
            q_resource = queue.pop()

            query = f"""
            PREFIX dbo: <http://dbpedia.org/ontology/>
            
            SELECT *
            FROM <{dbpedia.graph_iri}>
            WHERE {{
              OPTIONAL {{ <{q_resource}> dbo:subdivision ?o }}
              OPTIONAL {{ <{q_resource}> dbo:country ?o }}
            }}
            """
            results = dbpedia.query(query)

            # collect URIs of entities of which q_resource is a part of
            data = results["results"]["bindings"]
            q_dependencies = {row["o"]["value"] for row in data if row}
            dependencies[q_resource] = q_dependencies

            # add new URIs to queue
            visited.add(q_resource)
            queue += list(q_dependencies - visited)

        if len(dependencies) == 1:
            # we end up here if no hierarchy was found
            geo_hierarchy = None
        else:
            # The DBpedia part-of hierarchy isn't clean and contains circular references. As long as creating a
            # topological order of all locations fails, drop the location queried last (note that we always query from
            # most specific to least specific, so these should be unimportant late encounters we can drop without
            # losing too much info).
            geo_hierarchy = None
            while geo_hierarchy is None and dependencies:
                try:
                    geo_hierarchy = toposort_flatten(dependencies, sort=True)

                    # reverse it
                    geo_hierarchy = geo_hierarchy[::-1]
                except CircularDependencyError:
                    dependencies.popitem()

        ser = pd.Series({GEO_HIERARCHY: geo_hierarchy})
        return ser

    def _process_dataset(self,
                         dataset: Dataset,
                         live_objects: Dict) -> Dataset:
        dbpedia = live_objects[DBPEDIA] # type: DbPedia

        locations = dataset.mentions_location
        assert locations is not None and DBPEDIA_URI in locations.columns, "Need to entity link locations to DBpedia first!"

        linked_locations = locations.loc[locations[DBPEDIA_URI].notna(), DBPEDIA_URI]

        # look up coordinates, then reindex to make indices match
        tqdm.pandas(desc="Look up locations on DBpedia")
        with_coordinates = linked_locations.progress_apply(lambda uri: self._look_up_coordinates(uri, dbpedia))
        with_coordinates_reindexed = with_coordinates.reindex(locations.index)

        # look up geographic hierarchy, then reindex to make indices match
        tqdm.pandas(desc="Look up geographic hierarchy on DBpedia")
        with_hierarchy = linked_locations.progress_apply(lambda uri: self._look_up_geographic_hierarchy(uri, dbpedia))
        with_hierarchy_reindexed = with_hierarchy.reindex(locations.index)

        dataset.mentions_location = pd.concat([locations, with_coordinates_reindexed, with_hierarchy_reindexed], axis=1)
        return dataset


component = LocationEnhanceStage