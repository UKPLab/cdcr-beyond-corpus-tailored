import datetime
from collections import Counter
from typing import Dict, Optional

import pandas as pd
from dateutil.relativedelta import relativedelta

from python import MENTION_TYPE, TIME_OF_THE_DAY, TIME_DATE
from python.handwritten_baseline import TIMEX_NORMALIZED, TIMEX_NORMALIZED_PARSED
from python.handwritten_baseline.pipeline.data.base import BaselineDataProcessorStage, Dataset


class TimexParsingStage(BaselineDataProcessorStage):
    """
    Converts grounded TIMEX strings into datetime objects. This is necessary because there are TIMEX expressions
    involving weeks, seasons, things like "NI" for night or "AF" for afternoon which to_datetime() does not handle for
    us.
    """
    # TODO complete this with info from http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.454.3264&rep=rep1&type=pdf
    #   page 60ff

    SEASON_MAP = {
        "WI": relativedelta(months=0, weeks=2),  # aim for some time mid-January (northern hemisphere only...)
        "SP": relativedelta(months=4, weeks=2),  # mid-April
        "SU": relativedelta(months=7, weeks=2),  # mid-July
        "FA": relativedelta(months=10, weeks=2),  # mid-October
    }

    TIME_OF_DAY_MAP = {
        "MO": datetime.timedelta(hours=9, minutes=30),  # morning is 09:30 AM
        "DT": datetime.timedelta(hours=13),  # let daytime be 1PM
        "AF": datetime.timedelta(hours=15),  # afternoon is 3PM
        "EV": datetime.timedelta(hours=19),  # evening is 7PM
        "NI": datetime.timedelta(hours=3),  # night is 3AM
    }

    def __init__(self, pos, config, config_global, logger):
        super(TimexParsingStage, self).__init__(pos, config, config_global, logger)

    def timex_to_datetime(self, s: str) -> Optional[datetime.datetime]:
        """
        Convert special TIMEX formats into datetimes.
        :param s: TIMEX string
        :return: datetime or None
        """
        if "REF" in s:
            # PAST_REF ("once"), PRESENT_REF ("now"), etc.
            return None
        elif "INTERSECT" in s:
            # 2013-10-25-WXX-5 INTERSECT P1D
            return None
        elif s.startswith("P"):
            # sometimes, period strings like "P1D-#1" make it in here
            return None
        elif "X" in s:
            # we drop placeholders which are not resolved at this point
            self.logger.warning(f"Could not parse '{s}'.")
            return None
        elif not "DT" in s and Counter(s)["T"] > 1:
            # we drop ranges like "2014-04-02T17:00-2015-03-10T02:35"
            self.logger.warning(f"Could not parse '{s}'.")
            return None

        if "T" in s:
            t_marker = s.index("T")
            datestr = s[:t_marker]
            modifierstr = s[t_marker + 1:]
        else:
            datestr = s
            modifierstr = None
        datestr_parsed = None
        delta = datetime.timedelta()

        # look for weeks
        if any(season in datestr for season in TimexParsingStage.SEASON_MAP.keys()):
            # go for seasons
            year, season = datestr.split("-")
            datestr_parsed = datetime.datetime(year=int(year), month=1, day=1) + TimexParsingStage.SEASON_MAP[season]
        elif "W" in datestr:
            # look for weekends
            if datestr.endswith("WE"):
                datestr = datestr[:datestr.rindex("-")] + "-7"
            else:
                # use the middle of the week (Wednesday 12 o'clock) for week dates, see also
                datestr = datestr + "-3"
                delta += datetime.timedelta(hours=12)

            # see https://stackoverflow.com/a/17087427
            try:
                datestr_parsed = datetime.datetime.strptime(datestr, "%G-W%V-%u")
            except ValueError:
                self.logger.warning(f"Could not parse '{s}'.")
                return None
        else:
            # try oldschool date format parsing
            for pattern in ["%Y-%m-%d", "%Y-%m", "%Y"]:
                try:
                    datestr_parsed = datetime.datetime.strptime(datestr, pattern)
                    break
                except ValueError:
                    continue
            if datestr_parsed is None:
                self.logger.warning(f"Could not parse '{s}'.")
                return None

        if modifierstr is not None:
            # try parsing times of the day
            if modifierstr in TimexParsingStage.TIME_OF_DAY_MAP:
                delta = TimexParsingStage.TIME_OF_DAY_MAP[modifierstr]
            else:
                # try oldschool time parsing
                assert ":" in modifierstr
                hours, minutes = modifierstr.split(":")
                delta = datetime.timedelta(hours=int(hours), minutes=int(minutes))

        datestr_parsed += delta
        return datestr_parsed

    def _process_dataset(self,
                         dataset: Dataset,
                         live_objects: Dict) -> Dataset:
        if dataset.mentions_time is None:
            self.logger.warning(
                "The given dataset does not contain any temporal mentions. Will return the dataset unprocessed.")

        mentions_time = dataset.mentions_time  # type: pd.DataFrame
        assert TIMEX_NORMALIZED in mentions_time.columns

        timex_normalized = mentions_time[TIMEX_NORMALIZED]

        # convert all timex expressions where possible
        timex_datetimed = pd.to_datetime(timex_normalized, errors="coerce")     # type: pd.Series

        # single out successfully parsed datetimes, and remove pesky timezone information
        parsed_with_pandas = timex_datetimed.loc[timex_datetimed.notna()].map(lambda v: v.replace(tzinfo=None))

        # apply custom TIMEX to datetime conversion for those mentions which (1) have a TIMEX expression that (2) failed
        # to parse with the standard method and which is (3) of a relevant mention type
        needs_custom_conversion = timex_normalized.notna() & timex_datetimed.isna() & mentions_time[MENTION_TYPE].isin([TIME_OF_THE_DAY, TIME_DATE])
        parsed_custom = timex_normalized.loc[needs_custom_conversion].map(lambda v: self.timex_to_datetime(v))

        # combine parsed results
        parsed_all = pd.concat([parsed_with_pandas, parsed_custom]).reindex(timex_normalized.index).sort_index()
        dataset.mentions_time[TIMEX_NORMALIZED_PARSED] = parsed_all
        return dataset


component = TimexParsingStage