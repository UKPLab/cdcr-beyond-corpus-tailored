import datetime
import hashlib
import logging
import pprint
import sys
from pathlib import Path
from typing import Optional

import diskcache
from ruamel.yaml import YAML
from ruamel.yaml.compat import StringIO


class YAMLWithStrDump(YAML):
    """See https://yaml.readthedocs.io/en/latest/example.html#output-of-dump-as-a-string"""

    def dump(self, data, stream=None, **kw):
        inefficient = False
        if stream is None:
            inefficient = True
            stream = StringIO()
        YAML.dump(self, data, stream, **kw)
        if inefficient:
            return stream.getvalue()


def get_filename_safe_string(s):
    keep_chars = "."
    return "".join([c if c.isalnum() or c in keep_chars else "_" for c in s]).strip()


def get_logger(config):
    logger = logging.getLogger("logger")
    formatter = logging.Formatter('%(threadName)s - %(asctime)s - %(levelname)s - %(module)s - %(message)s')
    handler_stdout = logging.StreamHandler(sys.stdout)
    handler_stdout.setLevel(logging.DEBUG)
    handler_stdout.setFormatter(formatter)
    logger.addHandler(handler_stdout)

    if 'path' in config:
        handler_file = logging.FileHandler(config['path'])
        handler_file.setLevel(config['level_file'])
        handler_file.setFormatter(formatter)
        logger.addHandler(handler_file)

    logger.setLevel(config.get('level_console', "DEBUG"))
    return logger


def get_dict_hash(d, shorten: bool = True):
    """
    Create string that uniquely identifies the given dict
    :param d:
    :param shorten: if `True`, will return only the first 8 characters of the hash
    :return:
    """
    # pretty print (necessary to keep a consistent order for dictionary entries, otherwise hash varies for same config), then md5 hash and keep first 8 chars
    hash = hashlib.md5(pprint.pformat(d).encode('utf-8')).hexdigest()
    return hash[:8] if shorten else hash


class YamlDisk(diskcache.Disk):
    """
    Loosely based off of http://www.grantjenks.com/docs/diskcache/tutorial.html#tutorial-disk
    """

    def __init__(self, *args, **kwargs):
        super(YamlDisk, self).__init__(*args, **kwargs)
        self.yaml = YAMLWithStrDump(typ="unsafe")

    def put(self, key):
        return super(YamlDisk, self).put(self.yaml.dump(key))

    def get(self, key, raw):
        data = super(YamlDisk, self).get(key, raw)
        return self.yaml.load(data)

    def store(self, value, read, key=diskcache.core.UNKNOWN):
        if not read:
            value = self.yaml.dump(value)
        return super(YamlDisk, self).store(value, read, key)

    def fetch(self, mode, filename, value, read):
        data = super(YamlDisk, self).fetch(mode, filename, value, read)
        if not read:
            data = self.yaml.load(data)
        return data


def get_date_based_subdirectory(dir: Optional[Path]):
    if dir is not None:
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        return dir / now
    else:
        return None
