
import pytest
from datetime import datetime, timedelta
from acoio.aco import ACOio, ACOLoader
from pathlib import Path

resources = Path("__file__").absolute().parent


@pytest.mark.commit
def mem_check():
    loader = ACOio(resources, ACOLoader)
    # select a date
    target = datetime(day=26, month=12, year=2012)

    # if needed, select a duration (defaults to 5 minutes)
    dur = timedelta(seconds=20)
    src = loader.load(target, dur)

    segment = src[: timedelta(seconds=6)]
    fs = 3200

    tst = loader.load(target + timedelta(minutes=3), timedelta(minutes=3))
    tst.highpass().lowpass().resample_fs(fs).subtract(segment.resample_fs(fs))
