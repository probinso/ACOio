from aco import ACOio, datetime, timedelta, _DatetimeACOLoader
import pydub
import os.path as osp
import os
import ray


ray.init()

def gentime(start, stop, step):
    target = start
    while target < stop:
        yield target
        target += step


@ray.remote
def func(target):
    loader = ACOio('/media/research/raw/')
    extension = 'mp3'

    durration = timedelta(minutes=5)
    dstdir = '/media/research/mp3/long2/'

    srcpath = _DatetimeACOLoader.path_from_date(target, None)

    fname, _extenasion = srcpath.rsplit('.', 1)
    dstpath = osp.join(dstdir, '.'.join([fname, extension]))
    os.makedirs(osp.dirname(dstpath), exist_ok=True)

    src = loader.load(target, durration)
    wav = src.get_wav(resample=False)

    sound = pydub.AudioSegment.from_wav(wav)
    sound.export(dstpath, format=extension)

    src = loader.load(target, durration)
    target = target + step


if __name__ == '__main__':
    step = timedelta(hours=1)
    start_date = datetime(
        month=1, year=2015, day=1
    )
    end_date = datetime(
        month=1, year=2016, day=1
    )

    li = list(target for target in gentime(start_date, end_date, step))
    results = [func.remote(i) for i in li]
    print(ray.get(results))

