import datetime
from logging import Logger
from pathlib import Path
from typing import Callable

import numpy as np
from optuna import Study
from optuna._study_direction import StudyDirection
from optuna.trial import FrozenTrial


class EarlyStoppingCallback(Callable):
    """
    Stops optimization if there is no improvement by a specifiable margin within a given number of trials.
    """

    def __init__(self,
                 logger: Logger,
                 patience: int = 10,
                 min_delta: float = 1e-4):
        """

        :param patience: number of trials to wait for improvement
        :param min_delta: minimum delta in value necessary to reset the trial counter
        """
        self.patience = patience
        self.min_delta = min_delta
        self.logger = logger

        self.best = None
        self.wait = 0

    def __call__(self, *args, **kwargs):
        study, trial = args     # type: Study, FrozenTrial

        if study.direction == StudyDirection.MAXIMIZE:
            compare_op = np.greater
            if self.best is None:
                self.best = -np.Inf
        elif study.direction == StudyDirection.MINIMIZE:
            compare_op = np.less
            if self.best is None:
                self.best = np.Inf
        else:
            raise ValueError

        if compare_op(trial.value - self.min_delta, self.best):
            self.best = trial.value
            self.wait = 0
            self.logger.debug("New optimum, resetting counter")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.logger.info(f"No improvement after {self.patience} trials. Stopping optimization.")
                study.stop()


class PlotCallback(Callable):
    """
    Creates a plot of the current optimization status each n trials.
    """

    def __init__(self,
                 serialization_dir: Path,
                 plot_every_n_trials: int = 10):
        """

        :param serialization_dir: plot destination
        :param plot_every_n_trials:
        """
        self.serialization_dir = serialization_dir
        self.plot_every_n_trials = plot_every_n_trials

        self.serialization_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_plot(study: Study, plot_title: str, output_file: Path):
        WAS_NEW_OPTIMUM_ONCE = "was_new_optimum_once"
        DATETIME_COMPLETE = "datetime_complete"
        VALUE = "value"
        SECONDS_ELAPSED = "seconds-elapsed"

        df = study.trials_dataframe()[[DATETIME_COMPLETE, VALUE]].sort_values(by=DATETIME_COMPLETE)

        # we want to show experiments which were the new best optimum results in a different color
        if study.direction == StudyDirection.MAXIMIZE:
            cumu = df[VALUE].cummax()
        else:
            cumu = df[VALUE].cummin()
        cumu_dedup = cumu.drop_duplicates()
        df.loc[cumu_dedup.index, WAS_NEW_OPTIMUM_ONCE] = "yes"
        df[WAS_NEW_OPTIMUM_ONCE] = df[WAS_NEW_OPTIMUM_ONCE].fillna("no").map({"yes": "red", "no": "gray"})

        # convert to seconds elapsed to have a numeric x axis for the scatter plot
        df[SECONDS_ELAPSED] = (df[DATETIME_COMPLETE] - df[DATETIME_COMPLETE].min()).dt.total_seconds()

        ax = df.plot.scatter(x=SECONDS_ELAPSED, y=VALUE, c=df[WAS_NEW_OPTIMUM_ONCE])
        ax.set_xlim(left=min(df[SECONDS_ELAPSED]), right=max(df[SECONDS_ELAPSED]))
        fig = ax.get_figure()
        fig.suptitle(plot_title)
        fig.savefig(str(output_file))

    def __call__(self, *args, **kwargs):
        study, trial = args  # type: Study, FrozenTrial

        if (trial.number + 1) % self.plot_every_n_trials == 0:
            date_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            plot_title = f"Optimization status after {trial.number + 1} trials at {date_string}"
            plot_destination = self.serialization_dir / f"{date_string}_{trial.number + 1}_trials_{study.best_value}_best.png"
            self.make_plot(study, plot_title, plot_destination)