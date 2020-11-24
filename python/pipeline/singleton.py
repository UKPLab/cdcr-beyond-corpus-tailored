from collections import MutableMapping

import ruamel.yaml

from python.common_components import CORENLP, YAML, DBPEDIA, DBPEDIA_SPOTLIGHT
from python.common_components.dbpedia import DbPedia
from python.common_components.dbpedia_spotlight import DbPediaSpotlight
from python.pipeline import ComponentBase
from python.common_components.corenlp import CoreNlp


class LiveObjectsSingletonDict(MutableMapping, ComponentBase):
    """
    During the pipeline execution, pipelines can pass on python objects by means of a dict. This class functions like a dict, but additionally provides singleton instances of relevant objects (mturk, mace, ...) to pipeline stages. In order to do that, the __get_item__ method intercepts lookups to predefined string constants.
    """

    def __init__(self, config, config_global, logger):
        super(LiveObjectsSingletonDict, self).__init__(config, config_global, logger)
        self.store = dict()

        self.generic_components = {CORENLP: CoreNlp, DBPEDIA: DbPedia, DBPEDIA_SPOTLIGHT: DbPediaSpotlight}

        self.keywords = list(self.generic_components.keys()) + [YAML]

    def _instantiate_singleton(self, item):
        if item == YAML:
            obj = ruamel.yaml.YAML(typ='unsafe')
            import warnings
            warnings.simplefilter('ignore', ruamel.yaml.error.UnsafeLoaderWarning)
        elif item in self.generic_components:
            obj = self.generic_components[item](self.config.get(item, {}), self.config_global, self.logger)
        else:
            raise ValueError
        self.store[item] = obj

    def __getitem__(self, item):
        if item in self.keywords and not item in self.store:
            self._instantiate_singleton(item)

        return self.store[item]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __repr__(self):
        return repr(sorted(self.items()))