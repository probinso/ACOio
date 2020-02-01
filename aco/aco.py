from datetime import datetime, timedelta
import os.path as osp
import re
import warnings
from operator import attrgetter
from functools import reduce

import numpy as np

from memoized_property import memoized_property
import pydub

from .sound import Sound


class _FileLoader:
    resolution = np.int32
    time_code = '%Y-%m-%d--%H.%M'

    @classmethod
    def load_ACO_from_file(cls, basedir, relpath):
        time_stamp = cls._date_from_filename(relpath)
        filename = osp.join(basedir, relpath)
        fs = cls._frames_per_second(filename)
        data = cls._data_from_file(filename)
        return ACO(time_stamp, fs, data, True)

    @classmethod
    def _data_from_file(cls, filename):
        raise NotImplementedError

    @classmethod
    def _date_from_filename(cls, filename):
        # 2016-02-15--05.00.HYD24BBpk
        name = osp.basename(filename)
        dts, _ = name.rsplit('.', 1)
        time_stamp = datetime.strptime(dts, cls.time_code)
        return time_stamp

    def _frames_per_second(cls, filename):
        raise NotImplementedError


class _FileMp3Loader(_FileLoader):
    extension = 'mp3'

    @classmethod
    def _data_from_file(cls, filename):
        a = pydub.AudioSegment.from_mp3(filename)
        y = np.array(a.get_array_of_samples())
        return y

    @classmethod
    def _frames_per_second(cls, filename):
        a = pydub.AudioSegment.from_mp3(filename)
        return a.frame_rate


class _FileACOLoader(_FileLoader):
    extension = 'HYD24BBpk'
    header_dtype = np.dtype(
        [('Record', '<u4'),
         ('Decimation', '<u1'),
         ('StartofFile', '<u1'),
         ('Sync1', '<u1'),
         ('Sync2', '<u1'),
         ('Statusbyte1', '<u1'),
         ('Statusbyte2', '<u1'),
         ('pad1', '<u1'),
         ('LeftRightFlag', '<u1'),
         ('tSec', '<u4'),
         ('tuSec', '<u4'),
         ('timecount', '<u4'),
         ('Year', '<i2'),
         ('yDay', '<i2'),
         ('Hour', '<u1'),
         ('Min', '<u1'),
         ('Sec', '<u1'),
         ('Allignment', '<u1'),
         ('sSec', '<i2'),
         ('dynrange', '<u1'),
         ('bits', '<u1')])

    @classmethod
    def _ACO_to_int(cls, databytes, nbits):
        '''
        Convert the block of bytes to an array of int32.

        We need to use int32 because there can be 17 bits.
        '''
        nbits = int(nbits)
        # Fast path for special case of 16 bits:
        if nbits == 16:
            return databytes.view(np.int16).astype(cls.resolution)
        # Put the bits in order from LSB to MSB:
        bits = np.unpackbits(databytes).reshape(-1, 8)[:, ::-1]
        # Group by the number of bits in the int:
        bits = bits.reshape(-1, nbits)
        # Reassemble the integers:
        pows = 2 ** np.arange(nbits, dtype=cls.resolution)
        num = (bits * pows).sum(axis=1).astype(cls.resolution)
        # Handle twos-complement negative integers:
        neg = num >= 2**(nbits-1)
        num[neg] -= 2**nbits
        return num

    @classmethod
    def _frames_per_second(cls, path):
        name = osp.basename(path)
        _, encs = name.rsplit('.', 1)
        fs = int(re.findall('\d+', encs).pop()) * 1000
        return fs

    @classmethod
    def _params_from_filename(cls, filename):
        timestamp = cls._date_from_filename(filename)
        fs = cls._frames_per_second(filename)
        return timestamp, fs

    @classmethod
    def _data_from_file(cls, filename):
        headerlist = []
        datalist = []
        with open(filename, 'rb') as fid:
            fid.seek(0, 2)
            eof = fid.tell()
            fid.seek(0, 0)
            while fid.tell() < eof:
                header = np.fromfile(fid, count=1, dtype=cls.header_dtype)[0]
                headerlist.append(header)
                nbits = int(header['bits'])
                count = (4096//8) * nbits
                databytes = np.fromfile(fid, count=count, dtype='<u1')
                data = cls._ACO_to_int(databytes, nbits)
                datalist.append(data)

        # headers = np.array(headerlist)
        # Keeping the blocks separate, matching the headers:
        data = np.vstack(datalist)

        # But we can also view it as a single time series:
        alldata = data.reshape(-1)
        return alldata


class _DatetimeLoader:
    expected_file_length = timedelta(minutes=5)

    @classmethod
    def __floor_dt(cls, dt):
        src = timedelta(hours=dt.hour, minutes=dt.minute, seconds=dt.second)
        offset = src.total_seconds() % cls.expected_file_length.total_seconds()
        return dt - timedelta(seconds=offset)

    @classmethod
    def _filename_from_date(cls, index_datetime):
        dts = datetime.strftime(index_datetime, cls.time_code)
        return '.'.join([dts, cls.extension])

    @classmethod
    def _path_from_date(cls, index_datetime, normdir):
        info = [index_datetime.year, index_datetime.month, index_datetime.day]
        dirname = osp.join(*map(lambda i: str(i).zfill(2), info)) \
            if normdir is None else normdir
        basename = cls._filename_from_date(index_datetime)
        return osp.join(dirname, basename)

    @classmethod
    def path_from_date(cls, search_datetime, normdir=None):
        floor_datetime = cls.__floor_dt(search_datetime)
        return cls._path_from_date(floor_datetime, normdir=normdir)

    @classmethod
    def _load_full_ACO_from_base_datetime(
        cls,
        basedir, floor_datetime
    ):
        aco = cls.load_ACO_from_file(
            basedir, cls._path_from_date(floor_datetime)
        )
        return aco

    @classmethod
    def load_span_ACO_from_datetime(
        cls,
        basedir, index_datetime,
        durration
    ):
        result = []
        floor_datetime = cls.__floor_dt(index_datetime)
        start = index_datetime - floor_datetime
        end = start + durration
        local_end = end
        while local_end.total_seconds() > 0:
            try:
                _ = cls._load_full_ACO_from_base_datetime(
                    basedir,
                    floor_datetime
                )
            except FileNotFoundError:
                warnings.warn(
                    'index-range not continuous in local storage',
                    UserWarning
                )
                break
            floor_datetime = cls.__floor_dt(
                floor_datetime + start + cls.expected_file_length
            )
            local_end = local_end - _._durration
            result.append(_)
        aco = reduce(ACO.__matmul__, result).squash_nan()
        return aco[start:end]


class _DatetimeACOLoader(_DatetimeLoader, _FileACOLoader):
    pass


class _DatetimeMp3Loader(_DatetimeLoader, _FileMp3Loader):
    pass


class Loader:
    def __init__(self, basedir):
        self.basedir = basedir

    def _path_loader(self, target):
        raise NotImplementedError

    def _date_loader(self, target, durration):
        raise NotImplementedError

    def load(self, target, durration=None):
        if isinstance(target, str):
            return self._path_loader(target, durration)
        elif isinstance(target, datetime):
            if durration is None:
                durration = timedelta(minutes=5)
            return self._date_loader(target, durration)
        else:
            raise TypeError


class ACOLoader(Loader):
    def _path_loader(self, target):
        return _DatetimeACOLoader.load_ACO_from_file(self.basedir, target)

    def _date_loader(self, target, durration):
        return _DatetimeACOLoader.load_span_ACO_from_datetime(
            self.basedir, target, durration)


class Mp3Loader(Loader):
    def _path_loader(self, target):
        return _DatetimeMp3Loader.load_ACO_from_file(self.basedir, target)

    def _date_loader(self, target, durration):
        return _DatetimeMp3Loader.load_span_ACO_from_datetime(
            self.basedir, target, durration)


class ACOio:
    def __init__(self, basedir, Loader=ACOLoader):
        self.loader = Loader(basedir)

    def load(self, target, durration=None):
        return self.loader.load(target, durration)


class ACO(Sound):
    def __init__(self, time_stamp, fs, data, raw=False):
        super().__init__(fs, data)
        self.start_datetime = time_stamp
        self.raw = raw

    def copy(self):
        return ACO(
            self.start_datetime,
            self._fs,
            self._data.copy(),
            self.raw
        )

    @memoized_property
    def end_datetime(self):
        return self.date_offset(self._durration)

    def date_offset(self, durration):
        return self.start_datetime + durration

    def _date_difference(self, d):
        return self.durration_to_index(d - self.start_datetime)

    def __oolb(self, slice_):
        return (slice_.start < timedelta(0))

    def __ooub(self, slice_):
        return (self.date_offset(slice_.stop) > self.end_datetime)

    def _oob(self, slice_):
        return self.__oolb(slice_) or self.__oolb(slice_)

    @classmethod
    def _reversed_indexing(cls, slice_):
        return (slice_.stop < slice_.start)

    def __getitem__(self, slice_):
        idx = timedelta(seconds=0) if slice_.start is None else slice_.start
        jdx = self._durration if slice_.stop is None else slice_.stop
        slice_ = slice(idx, jdx)

        if self._reversed_indexing(slice_):
            raise "Does not support reverse indexing"

        if self._oob(slice_):
            warnings.warn(f'Slice Out of Bounds', UserWarning)

        result = self.copy()
        start = slice_.start
        timestamp = self.start_datetime + (
            timedelta(0) if start is None else start
        )

        idx, jdx = self._getitem__indicies(slice_)
        data = self._data[idx:jdx]
        result._data = data
        result.timestamp = timestamp
        return result

    def __matmul__(self, other):
        '''
        allows date-time respecting joins of tracks
        '''
        assert(self.raw)
        assert(other.raw)

        A, B = self.copy(), other.copy()

        ordered = (A, B)  # wlg
        if self._fs != other._fs:
            ordered = sorted((self, other), key=attrgetter('_fs'))
            ordered[-1] = ordered[-1].resample_fs(ordered[0]._fs)

        ordered = sorted(ordered, key=attrgetter('start_datetime'))
        durration = ordered[-1].end_datetime - ordered[0].start_datetime

        space = max(
            ordered[0].durration_to_index(durration),
            len(A._data), len(B._data))

        data = np.full(space, np.NAN)

        idx = ~np.isnan(ordered[0]._data)
        data[:len(ordered[0]._data)][idx] = ordered[0]._data[idx]

        durration = ordered[-1].start_datetime - ordered[0].start_datetime
        start = ordered[0].durration_to_index(durration)

        idx = ~np.isnan(ordered[-1]._data)
        overlap_count = np.sum(~np.isnan(data[start:][idx]))

        data[start:][idx] = ordered[-1]._data[idx]

        if overlap_count > 0:
            warnings.warn(f'Overlaps {overlap_count} samples', UserWarning)

        result = self.__class__(
            ordered[0].start_datetime,
            ordered[0]._fs,
            data,
            ordered[0].raw
        )
        return result


if __name__ == '__main__':
    loader = ACOio('./dst/', Mp3Loader)
    target = datetime(
        day=1, month=2, year=2016
    )
    aco = loader.load(target)
