
from acoio.aco import ACOio, ACOLoader, datetime, timedelta


def mem_check():
    loader = ACOio("/home/probinso/DATA/Whales", ACOLoader)
    # select a date
    target = datetime(day=1, month=1, year=2013)

    # if needed, select a duration (defaults to 5 minutes)
    dur = timedelta(seconds=20)
    src = loader.load(target, dur)

    segment = src[: timedelta(seconds=6)]
    fs = 3200

    tst = loader.load(target + timedelta(minutes=3), timedelta(minutes=3))
    tst.highpass().lowpass().resample_fs(fs).subtract(segment.resample_fs(fs))
