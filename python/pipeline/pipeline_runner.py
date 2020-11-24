import importlib
import sys

import click

from python.pipeline.common import CleanupStage
from python.util.config import load_config, set_up_dir_structure
from python.pipeline.singleton import LiveObjectsSingletonDict
from python.util.util import get_logger


@click.group()
@click.option("--debug", "debug", type=bool, default=False, help="Enable PyCharm remote debugger")
@click.option("--ip", "pycharm_debugger_ip", type=str, default=None, help="PyCharm debugger IP")
@click.option("--port", "pycharm_debugger_port", type=int, default=None, help="PyCharm debugger port")
def cli_pipeline_runner_init(debug, pycharm_debugger_ip, pycharm_debugger_port):
    # Before running any specific train/eval code, we start up the remote debugger if desired.
    if debug:
        try:
            # see https://www.jetbrains.com/help/pycharm/remote-debugging-with-product.html
            import pydevd_pycharm
            pydevd_pycharm.settrace(pycharm_debugger_ip, port=pycharm_debugger_port, stdoutToServer=True,
                                    stderrToServer=True)
        except ImportError as e:
            print("pydevd_pycharm is not installed. No remote debugging possible.")
            print(str(e))
            sys.exit(1)


@cli_pipeline_runner_init.command(help="Run the pipeline")
@click.argument("config_paths", nargs=-1, type=click.Path(exists=True, dir_okay=False))
def run(config_paths):
    _run(config_paths)

def _run(config_paths, **kwargs):
    """
    Run the pipeline
    :param config_paths:
    :param kwargs: the pipeline live objects will be initialized with this mapping
    :return: the content of live_objects after the last pipeline stage
    """
    if len(config_paths) == 0:
        print("Need at least one path to a config file.")
        sys.exit(1)

    config = load_config(config_paths)
    config = set_up_dir_structure(config)

    config_global = config["global"]

    # setup a logger
    logger = get_logger(config_global['logging'])
    logger.info("Creating pipeline...")

    # We are now fetching all relevant modules. It is strictly required that these module contain a variable named
    # 'component' that points to a class which inherits from ...
    pipeline_stage_descriptions = config['pipeline']

    # The modules are now dynamically loaded and connected
    pipeline_stages = []
    productions = {}
    for pos, stage_description in enumerate(pipeline_stage_descriptions):
        stage_module, stage_config = list(stage_description.items())[0]
        clazz = importlib.import_module(stage_module).component

        # use empty config if none is specified
        if stage_config is None:
            stage_config = {}

        stage = clazz(pos, stage_config, config_global, logger)

        # pass productions from previous stages to new pipeline stage, and add its productions for later stages
        stage.requires_files(productions)
        productions.update(stage.files_produced())
        pipeline_stages.append(stage)
    # always append cleanup stage
    pipeline_stages.append(CleanupStage(len(pipeline_stage_descriptions), {}, config_global, logger))

    logger.info(
        "Pipeline created: {}".format(" --> ".join([f"{p.position:02}__{p.short_name}" for p in pipeline_stages])))

    live_objects = LiveObjectsSingletonDict(config, config_global, logger)
    if kwargs:
        live_objects.update(kwargs)
    logger.info("Starting pipeline.")
    for stage in pipeline_stages:
        logger.info(f"Running {stage.name}")
        stage.run(live_objects)
    logger.info("Pipeline finished.")

    return live_objects