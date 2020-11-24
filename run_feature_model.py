import sys
import time
from pathlib import Path

import click

from python.handwritten_baseline.pipeline.model.scripts.baselines.lemma_baselines import run_baselines
from python.handwritten_baseline.pipeline.model.scripts.train_predict_optimize import \
    feature_selection as do_feature_selection, optimize_hyperparameters, train as do_train, evaluate as do_evaluate
from python.pipeline import JOB_ID_RAW, RUN_WORKING_DIR, MAX_CORES
from python.util.config import set_up_dir_structure, load_config, write_config
from python.util.util import get_logger


def _initialize(config_paths):
    if len(config_paths) == 0:
        print("Need at least one path to a config file.")
        sys.exit(1)

    config = load_config(config_paths)
    config = set_up_dir_structure(config)

    config_global = config["global"]
    serialization_dir = config_global[RUN_WORKING_DIR]

    # save config for reference for later
    write_config(config, serialization_dir / "config.yaml")

    # setup a logger
    logger = get_logger(config_global['logging'])

    # set random seeds
    if config_global["development_mode"]:
        logger.warning("\n"
                       "############## DEVELOPMENT MODE ##############\n"
                       "# fixed random seed and no parallelization!  #\n"
                       "##############################################")
        random_seed = 42
        config_global[MAX_CORES] = 1
    else:
        random_seed = config_global[JOB_ID_RAW] or time.time_ns() % 2 ** 32

    return config, logger, random_seed


@click.group()
@click.option("--debug", "debug", type=bool, default=False, help="Enable PyCharm remote debugger")
@click.option("--ip", "pycharm_debugger_ip", type=str, default=None, help="PyCharm debugger IP")
@click.option("--port", "pycharm_debugger_port", type=int, default=None, help="PyCharm debugger port")
def cli_init(debug, pycharm_debugger_ip, pycharm_debugger_port):
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


@cli_init.command(help="Identify most useful features")
@click.argument("config_paths", nargs=-1, type=click.Path(exists=True, dir_okay=False))
def feature_selection(config_paths):
    config, logger, _ = _initialize(config_paths)
    do_feature_selection(config["data"], config["global"], logger)


@cli_init.command(help="Run hyperparameter optimization")
@click.argument("config_paths", nargs=-1, type=click.Path(exists=True, dir_okay=False))
def hyperopt(config_paths):
    config, logger, random_seed = _initialize(config_paths)
    optimize_hyperparameters(config["data"],
                             config["model"],
                             config["hyperopt"],
                             config["global"],
                             logger)


@cli_init.command(help="Train a model")
@click.argument("config_paths", nargs=-1, type=click.Path(exists=True, dir_okay=False))
def train(config_paths):
    config, logger, random_seed = _initialize(config_paths)
    do_train(config["data"],
             config["model"],
             config["training"],
             config["global"],
             logger)


@cli_init.command(help="Evaluate a model")
@click.argument("model_serialization_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("config_paths", nargs=-1, type=click.Path(exists=True, dir_okay=False))
def evaluate(model_serialization_dir, config_paths):
    config, logger, random_seed = _initialize(config_paths)
    model_serialization_dir = Path(model_serialization_dir)
    do_evaluate(model_serialization_dir,
                config["data"],
                config["evaluate"],
                config["global"],
                logger)


@cli_init.command(help="Run lemma baselines")
@click.argument("config_paths", nargs=-1, type=click.Path(exists=True, dir_okay=False))
def lemma_baselines(config_paths):
    config, logger, random_seed = _initialize(config_paths)
    run_baselines(config["data"],
                  config["baselines"],
                  config["global"],
                  logger)


if __name__ == "__main__":
    cli_init()
