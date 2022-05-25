import os
from pathlib import Path

import nltk

from python.pipeline import ComponentBase


class Nltk(ComponentBase):

    def __init__(self, config, config_global, logger):
        super(Nltk, self).__init__(config, config_global, logger)

        nltk_data = config.get("data_dir", None)
        if nltk_data:
            # resolve nltk_data against the shell's working dir
            os.environ["NLTK_DATA"] = str(Path.cwd() / Path(nltk_data))

        for _id, resulting_path in config.get("dependencies", {}).items():
            try:
                nltk.data.find(resulting_path)
            except LookupError:
                nltk.download(_id)