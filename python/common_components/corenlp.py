import os
from pathlib import Path
from typing import Optional, Dict
from python.util.util import get_dict_hash

from overrides import overrides
from stanfordnlp.protobuf import Document, parseFromDelimitedString, writeToDelimitedString
from stanfordnlp.server import CoreNLPClient

from python.pipeline import ComponentBase


class CoreNlp(ComponentBase):

    def __init__(self, config, config_global, logger):
        super(CoreNlp, self).__init__(config, config_global, logger)

        self.cache = self._provide_cache("stanfordnlp_cache", human_readable=False)

        corenlp_home = config.get("corenlp_home", None)
        if corenlp_home:
            # resolve corenlp_home against the shell's working dir
            os.environ["CORENLP_HOME"] = str(Path.cwd() / Path(corenlp_home))

        self._kwargs = config.pop("corenlp_kwargs", {"annotators": "depparse"})
        self._client = None  # type: Optional[CoreNLPClient]

    def parse_sentence(self, sentence: str, properties: Optional[Dict] = None):
        """
        Run CoreNLP over a sentence.
        :param sentence: a single sentence
        :param properties: additional properties for CoreNLP
        :return: parsing result
        """
        # The same input sentence can result in different annotations depending on the CoreNLP properties specified.
        # We therefore use a cache identifier for the sentence which includes the annotation properties.
        sent_cache_identifier = get_dict_hash({"sentence": sentence, "properties": properties}, shorten=False)

        if not sent_cache_identifier in self.cache:
            # Kludge ahead: We want to cache the parsed sentence provided by CoreNLP, but also want to work with it in
            # a convenient format. A convenient format is the default format (protobuf-based), but that's not
            # pickle-able for the cache. We therefore convert the protobuf-format back into a bytestring and cache that.
            # When reading from the cache, we reassemble the protobuf object.
            req_properties = {"outputFormat": "serialized"}
            if properties is not None:
                req_properties.update(properties)
            doc = self.client.annotate(sentence, properties=req_properties)
            stream = writeToDelimitedString(doc)
            buf = stream.getvalue()
            stream.close()
            self.cache[sent_cache_identifier] = buf
        else:
            buf = self.cache[sent_cache_identifier]
            doc = Document()
            parseFromDelimitedString(doc, buf)

        return doc

    @property
    def client(self):
        if self._client is None:
            self._client = CoreNLPClient(**self._kwargs)
            self._client.start()
        return self._client

    @overrides
    def clean_up(self):
        if self._client is not None:
            self._client.stop()
