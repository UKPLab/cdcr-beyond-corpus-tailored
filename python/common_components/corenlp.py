import itertools
import time
import os
from pathlib import Path
from typing import Optional, Dict, Iterable, Tuple, Union

from joblib import delayed, Parallel
from more_itertools import take
from overrides import overrides
from stanfordnlp.protobuf import Document, parseFromDelimitedString, writeToDelimitedString
from stanfordnlp.server import CoreNLPClient

from python.pipeline import ComponentBase, MAX_CORES, RUN_TEMP
from python.util.util import get_dict_hash


class CoreNlp(ComponentBase):

    def __init__(self, config, config_global, logger):
        super(CoreNlp, self).__init__(config, config_global, logger)

        self.cache = self._provide_cache("stanfordnlp_cache", human_readable=False, scope=RUN_TEMP)

        corenlp_home = config.get("corenlp_home", None)
        if corenlp_home:
            # resolve corenlp_home against the shell's working dir
            os.environ["CORENLP_HOME"] = str(Path.cwd() / Path(corenlp_home))
        kwargs = config.pop("corenlp_kwargs", {"annotators": "depparse"})
        kwargs["threads"] = max(1, config_global[MAX_CORES] - 1)
        self._kwargs = kwargs

        self._client = None  # type: Optional[CoreNLPClient]

        # number of threads of the remote CoreNLP server (if used), this determines the batch size for parallel
        # annotation here
        self._server_threads = config.pop("server_threads", 1)
        assert self._server_threads >= 1

    @staticmethod
    def _get_identifier(s: str, properties: Optional[Dict] = None) -> str:
        # The same input sentence can result in different annotations depending on the CoreNLP properties specified.
        # We therefore use a cache identifier for the sentence which includes the annotation properties.
        return get_dict_hash({"sentence": s, "properties": properties}, shorten=False)

    @staticmethod
    def _extend_corenlp_properties(properties: Optional[Dict] = None):
        # properties actually used for requests; i.e. more technical bits not relevant for our cache
        req_properties = {"outputFormat": "serialized",
                          "serializer": "edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer"}
        if properties is not None:
            req_properties.update(properties)
        return req_properties

    @staticmethod
    def _patient_request(client: CoreNLPClient, s: str, p: Dict) -> Tuple[Document, Optional[Exception]]:
        """
        Issues arise with long-running experiments which use a remote CoreNLP service that restarts during the
        experiment (because the CoreNLP service is subject to a maximum job runtime on Slurm for example). In case of
        annotation errors (any), this method waits 45 seconds then tries again once. If unsuccessful, returns the
        exception from the last attempt.
        :param client: CoreNLP client
        :param s: string to annotate
        :param p: annotation properties
        """
        # 45 seconds are long, but annotation errors outside of disconnects are rare, so this should be fine
        patience_seconds = 45
        retries_left = 1
        exc = None
        while retries_left >= 0:
            try:
                return client.annotate(s, properties=p), None
            except Exception as e:  # ConnectionError, TimeoutException, AnnotationException, ...
                retries_left -= 1
                exc = e
                time.sleep(patience_seconds)
        return None, exc

    def _write_to_cache(self, doc: Document, cache_identifier: str):
        # Kludge ahead: We want to cache the annotations provided by CoreNLP, but also want to work with it in
        # a convenient format. A convenient format is the default format (protobuf-based), but that's not
        # pickle-able for the cache. We therefore convert the protobuf-format back into a bytestring and cache that.
        # When reading from the cache, we reassemble the protobuf object.
        stream = writeToDelimitedString(doc)
        buf = stream.getvalue()
        stream.close()
        self.cache[cache_identifier] = buf

    def _read_from_cache(self, cache_identifier: str) -> Document:
        buf = self.cache[cache_identifier]
        doc = Document()
        parseFromDelimitedString(doc, buf)
        return doc

    def parse_sentence(self, s: str, properties: Optional[Dict] = None, use_cache: bool = True) -> Tuple[Document, Optional[Exception]]:
        """
        Run CoreNLP on a string.
        :param s: a string (sentence, document, ...)
        :param properties: additional properties for CoreNLP
        :param use_cache:
        :return: parsing result
        """
        def do_parse():
            req_properties = CoreNlp._extend_corenlp_properties(properties)
            return CoreNlp._patient_request(self.client, s, req_properties)

        if not use_cache:
            return do_parse()
        else:
            cache_identifier = CoreNlp._get_identifier(s, properties)
            if not cache_identifier in self.cache:
                doc, exc = do_parse()
                if exc is not None:
                    self._write_to_cache(doc, cache_identifier)
            else:
                doc = self._read_from_cache(cache_identifier)
                exc = None
            return doc, exc

    def parse_strings(self,
                      strings: Iterable[str],
                      properties: Optional[Union[Iterable[Dict], Dict]] = None,
                      use_cache: bool = True) -> Iterable[Document]:
        """
        Returns iterator over annotated strings, where the annotations are performed in batches on a remote CoreNLP
        server. If this CoreNlp is not configured to use a remote server, falls back to non-parallelized local
        processing.
        """
        # make sure `properties_iter` is a useful iterator for CoreNLP request properties, whatever the input type
        if type(properties) is dict:
            properties_iter = itertools.repeat(properties)
        else:
            try:
                iter(properties)
                properties_iter = properties
            except TypeError as e:
                raise ValueError(f"{str(properties)} is not an accepted dictionary of properties, or iterable of dicts of properties.", e)

        is_using_remote_server = self.client.server is None
        if not is_using_remote_server:
            for s, p in zip(strings, properties_iter):
                doc, exc = self.parse_sentence(s, p, use_cache=use_cache)
                if exc is not None:
                    yield doc
            return

        # To iteratively parse input strings, and to still use our diskcache, this method uses a producer/consumer
        # setup involving iterators and a shared buffer. The buffer keeps (int, Document) tuples where the int is the
        # sequence position in the original list of input strings. This is so that we can return annotations in the
        # order of inputs.
        buffer = {} # type: Dict[int, Document]

        def produce_strings_to_parse():
            """
            Returns an iterator over strings we still need to annotate with CoreNLP. For input strings which were
            annotated previously (i.e. their annotations are cached), we read the annotation and put it in the buffer.
            """
            for i, (s, p) in enumerate(zip(strings, properties_iter)):
                cache_identifier = get_dict_hash({"s": s, "properties": p}, shorten=False)
                if use_cache and cache_identifier in self.cache:
                    buffer[i] = self._read_from_cache(cache_identifier)
                else:
                    p_extended = CoreNlp._extend_corenlp_properties(p)
                    yield i, s, p_extended, cache_identifier

        def annotate(s: str, p: Dict, client_kwargs: Dict) -> Tuple[Document, Optional[Exception]]:
            client = CoreNLPClient(**client_kwargs)
            doc, exc = CoreNlp._patient_request(client, s, p)
            client.stop()
            return doc, exc

        # TODO it's unclear if the overhead for parallelization is worth it here
        producer = produce_strings_to_parse()
        producer_has_more_items = True
        i_next_to_yield = 0     # index of the next annotation to yield, following the original order of `strings`
        batch_size = self._server_threads
        with Parallel(n_jobs=batch_size, prefer="threads") as parallel:
            while producer_has_more_items or buffer:
                # if the next annotation is known, i.e. is in the buffer, pop from buffer and advance to next item
                if i_next_to_yield in buffer.keys():
                    yield buffer.pop(i_next_to_yield)
                    i_next_to_yield += 1
                elif producer_has_more_items: # otherwise, if we still parse strings on-demand, parse one batch
                    batch = take(batch_size, producer)

                    # As long as we receive as many elements from `producer` as we ask for, there should be more
                    # items to retrieve. One exception: if we didn't obtain any in this iteration, then
                    # batch_size % num_items == 0 and we exhausted the producer in the last iteration, so we bail out.
                    producer_has_more_items = len(batch) == batch_size
                    if len(batch) == 0:
                        continue

                    # annotate batch, cache results and put annotated documents into buffer
                    annotations = parallel(delayed(annotate)(s, p, self._kwargs) for _, s, p, _ in batch)
                    for (i, _, _, cache_identifier), (doc, exc) in zip(batch, annotations):
                        if use_cache and exc is None:
                            self._write_to_cache(doc, cache_identifier)
                        if exc is not None:
                            self.logger.warning("Failed to annotate string: ", exc)
                        buffer[i] = doc
                else:
                    raise RuntimeError(f"Could not parse string at position {i_next_to_yield}. CoreNLP annotation failure?")

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
