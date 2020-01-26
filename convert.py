from aco import ACOio, datetime, timedelta, _DatetimeACOLoader
import pydub

if __name__ == '__main__':
    loader = ACOio('./')
    extension = 'mp3'

    durration = timedelta(minutes=5)
    step = timedelta(hours=1)
    start_date = datetime(
        month=2, year=2016, day=1
    )
    end_date = datetime(
        month=2, year=2016, day=2
    )
    # end_date = datetime(
    #     month=3, year=2016, day=1
    # )

    target = start_date
    while target < end_date:
        srcpath = _DatetimeACOLoader.path_from_date(target)
        fname, _extenasion = srcpath.rsplit('.', 1)
        dstpath = '.'.join([fname, extension])
        src = loader.load(target, durration)
        wav = src.get_wav(resample=False)

        sound = pydub.AudioSegment.from_wav(wav)
        sound.export(dstpath, format=extension)

        src = loader.load(target, durration)
        target = target + step
