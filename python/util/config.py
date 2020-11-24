import time
from pathlib import Path

import slurmee
from ruamel.yaml import YAML

from python.pipeline import ID, TIMESTAMP, JOB_ID, TASK_ID, GLOBAL_WORKING_DIR, CONFIG_NAME_WORKING_DIR, \
    CONFIG_WORKING_DIR, RUN_WORKING_DIR, JOB_ID_RAW, MAX_CORES
from python.util.util import get_dict_hash, get_filename_safe_string


def load_config(config_paths):
    """All given configurations are merged into one. If a configuration entry appears in multiple files, the value from the rightmost config file in the list will be used.

    :param config_paths: paths to the yaml config files
    :return: the configuration dictionary
    :rtype: tuple
    """
    yaml = YAML()

    config = {}
    for path in config_paths:
        with open(path) as f:
            new_config = yaml.load(f)
        config = merge_dicts(config, new_config)
    return config


def write_config(config_dict, destination):
    yaml = YAML(typ="unsafe")
    with open(destination, 'w') as f:
        yaml.dump(config_dict, f)


def merge_dicts(dict_a, dict_b):
    """Performs a recursive merge of two dicts dict_a and dict_b, wheras dict_b always overwrites the values of dict_a

    :param dict_a: the first dictionary. This is the weak dictionary which will always be overwritten by dict_b (dict_a
                   therefore is a default dictionary type
    :param dict_b: the second dictionary. This is the strong dictionary which will always overwrite vales of dict_a
    :return:
    """
    merge_result = dict_a.copy()
    for key in dict_b:
        if key in dict_a and isinstance(dict_a[key], dict) and isinstance(dict_b[key], dict):
            merge_result[key] = merge_dicts(dict_a[key], dict_b[key])
        else:
            merge_result[key] = dict_b[key]
    return merge_result


def set_up_dir_structure(config):
    """
    Preparation steps to get a directory structure like the following:
    <working_dir>
    ├── <config_name>
    │   └── <config_hash>
    │       ├── 00__<pipeline_stage_name>
    │       ├── 01__<other_pipeline_stage_name>
    │       ├── ...
    │       └── <timestamp of run>
    │           └── event.log
    └── global
    :param config: yaml config
    :return:
    """
    config_id = get_dict_hash(config)

    # Load the configuration and store its own identifier and the current timestamp in the global config
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    config_global = config["global"]
    config_global[ID] = config_id
    config_global[TIMESTAMP] = timestamp

    # use specified working dir, resolved against the working directory of the shell
    working_dir = Path.cwd() / Path(config_global["working_dir"])

    # set up a shared working dir for all runs exists (for cached datasets etc.)
    global_working_dir = working_dir / "global"

    # set up a config-name (!) specific directory if a name is given - this is just for better grouping of runs
    config_name = config_global.get("config_name", "pipeline")
    config_name_working_dir = working_dir / get_filename_safe_string(config_name)

    # set up a config-specific directory
    config_working_dir = config_name_working_dir / config_id

    # set up a run-specific directory - some conditional changes depending on whether we run inside Slurm or not
    run_working_dir_filename_parts = []
    config_global[TASK_ID] = None
    config_global[JOB_ID] = None
    config_global[JOB_ID_RAW] = None
    slurm_job_id = slurmee.get_job_id()
    job_array_info = slurmee.get_job_array_info()

    # if running on slurm:
    if slurm_job_id is not None:
        # use as many CPU cores as there are available
        config_global[MAX_CORES] = slurmee.get_cpus_on_node()

        # if running inside job array
        if job_array_info is not None:
            root_job_id = job_array_info["array_job_id"]
            task_id = job_array_info["task_id"]
            run_working_dir_filename_parts += [str(root_job_id), f"{task_id:0>2}"]
            config_global[JOB_ID] = root_job_id
            config_global[TASK_ID] = task_id
            config_global[JOB_ID_RAW] = root_job_id + task_id
        else:
            run_working_dir_filename_parts += [str(slurm_job_id)]
            config_global[JOB_ID] = slurm_job_id
            config_global[JOB_ID_RAW] = slurm_job_id
    run_working_dir_filename_parts += [timestamp]
    run_working_dir_filename = "_".join(run_working_dir_filename_parts)
    run_working_dir = config_working_dir / run_working_dir_filename

    # create folders
    for directory in [global_working_dir, run_working_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    config_global[GLOBAL_WORKING_DIR] = global_working_dir
    config_global[CONFIG_NAME_WORKING_DIR] = config_name_working_dir
    config_global[CONFIG_WORKING_DIR] = config_working_dir
    config_global[RUN_WORKING_DIR] = run_working_dir

    # ...redirect destination for the logging file handler
    if "logging" not in config_global:
        config_global["logging"] = {}
    config_global["logging"]["path"] = run_working_dir / "log_events.log"

    return config