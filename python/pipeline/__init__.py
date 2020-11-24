import inspect
import json
import os
import shutil
import tempfile
from typing import Dict, Union, List

from diskcache import FanoutCache

from python.util.util import get_dict_hash, YamlDisk

# pipeline-specific constants
GLOBAL_WORKING_DIR = "global_working_dir"
CONFIG_NAME_WORKING_DIR = "config_name_working_dir"
CONFIG_WORKING_DIR = "config_working_dir"
RUN_WORKING_DIR = "run_working_dir"
JOB_ID = "job_id"
JOB_ID_RAW = "job_id_raw"
TASK_ID = "task_id"     # if using job arrays, JOB_ID will be the root job ID and task ID will be the array task id
TIMESTAMP = "_time"
ID = "_id"
MAX_CORES = "max_cores"

# scopes for disk locations
RUN_TEMP = "run_temp"
RUN = "run"
CONFIG_NAME = "config_name"
GLOBAL = "global"


class ComponentBase(object):
    def __init__(self, config, config_global, logger):
        """This is a simple base object for all experiment components

        :type config: dict
        :type config_global: dict
        :type logger: logging.Logger
        """
        self.config = config or dict()
        self.config_global = config_global or dict()
        self.logger = logger

        self.clear_caches = config.get("clear_caches", False)

    def _provide_disk_location(self, filename, scope=RUN, make_dir=False, remove_if_exists=False):
        """
        Returns an os.path object in a useful location. Works both for files and directories.
        :param filename: file or directory name
        :param scope: one of GLOBAL, CONFIG_NAME, RUN or RUN_TEMP. GLOBAL locations are persistent across different
        configurations and runs (useful for global caches). CONFIG_NAME locations are persistent across configs of the
        same name. RUN locations are specific per run, are shared among stages and will not be overwritten. RUN_TEMP
        locations are located in the system's temp directory, are specific per run and may be (but are not
        automatically) overwritten after the run finishes.
        :param make_dir: if True, will create an empty directory at the location right away
        :param remove_if_exists: self-explanatory
        :return: path
        """
        # determine general location of new file/directory
        locations_by_scope = {GLOBAL: self.config_global[GLOBAL_WORKING_DIR],
                              CONFIG_NAME: self.config_global[CONFIG_NAME_WORKING_DIR],
                              RUN: self.config_global[RUN_WORKING_DIR],
                              RUN_TEMP: tempfile.gettempdir()}


        if scope not in locations_by_scope:
            raise ValueError(f"Unknown scope {scope}")
        else:
            directory = locations_by_scope[scope]

        # create parent directory of new file/directory
        os.makedirs(directory, exist_ok=True)

        path = os.path.join(directory, filename)

        # remove if desired
        if remove_if_exists and os.path.exists(path):
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)

        # create directory right away if desired
        if make_dir:
            os.makedirs(path, exist_ok=True)

        return path

    def _provide_cache(self, name: str = None, bind_parameters: Union[Dict, List] = None, scope=GLOBAL,
                       human_readable=False, size_limit: int = 10737418245) -> FanoutCache:
        """
        Creates a disk cache.
        :param name: a descriptive name (what do you want to cache?)
        :param bind_parameters: The contents of the cache most likely depend on some parameters (n-gram size, or whatever). Pass these parameters here. Caches are kept separate for different parameter sets to avoid disaster.
        :param scope: If GLOBAL, cache will be persistent across executions of this framework. See _provide_disk_location for more options.
        :param human_readable: Use a human readable cache format (yaml).
        :param size_limit: maximum size in bytes (default: one gigabyte)
        :return:
        """
        cache_dirname_components = [name if name else "cache"]
        if bind_parameters:
            params_hash = get_dict_hash(bind_parameters)
            cache_dirname_components.append(params_hash)
        cache_dirname = "_".join(cache_dirname_components)

        # for debugging purposes, create a json dump of the parameters the cache is bound to
        if bind_parameters:
            params_dump_file = self._provide_disk_location(f"{cache_dirname}_params.json", scope=scope)
            with open(params_dump_file, "w") as f:
                json.dump(bind_parameters, f)

        cache_dir = os.path.join(self._provide_disk_location(self.name, scope=scope), cache_dirname)

        if human_readable:
            cache = FanoutCache(cache_dir, size_limit=size_limit, disk=YamlDisk)
        else:
            cache = FanoutCache(cache_dir, size_limit=size_limit)

        if self.clear_caches:
            cache.clear()

        return cache

    @property
    def name(self):
        """
        Returns name of own class as "foo.bar.ClassName"
        :return:
        """
        return ".".join([inspect.getmodule(self).__name__, self.short_name])

    @property
    def short_name(self):
        return self.__class__.__name__

    def clean_up(self):
        """
        Performs cleanup tasks on this component (close sockets, remove files, ...).
        :return:
        """
        pass