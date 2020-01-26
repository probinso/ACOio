from aco import ACOio, datetime, timedelta, _DatetimeACOLoader
import pydub
import os.path as osp
import os

def gentime(start, stop, step):
    target = start
    while target < stop:
        yield target
        target += step

if __name__ == '__main__':
    loader = ACOio('/media/research/raw/')
    extension = 'mp3'

    durration = timedelta(minutes=5)
    step = timedelta(hours=1)
    start_date = datetime(
        month=12, year=2012, day=1
    )
    end_date = datetime(
        month=5, year=2013, day=1
    )
    dstdir = '/media/research/mp3/long/'

    for target in gentime(start_date, end_date, step):
        srcpath = _DatetimeACOLoader.path_from_date(target)

        fname, _extenasion = srcpath.rsplit('.', 1)
        dstpath = osp.join(dstdir, '.'.join([fname, extension]))
        os.makedirs(osp.dirname(dstpath), exist_ok=True)

        src = loader.load(target, durration)
        wav = src.get_wav(resample=False)

        sound = pydub.AudioSegment.from_wav(wav)
        sound.export(dstpath, format=extension)

        src = loader.load(target, durration)
        target = target + step

