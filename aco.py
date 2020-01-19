from datetime import datetime, timedelta
import os.path as osp
import re
import warnings
from operator import attrgetter
from functools import reduce

import numpy as np

from memoized_property import memoized_property


from sound import Sound


class _ACOLoader:
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

    resolution = np.int32
    time_code = '%Y-%m-%d--%H.%M'

    @classmethod
    def load_ACO_from_file(cls, basedir, relpath):
        time_stamp, fs = cls._params_from_filename(relpath)
        filename = osp.join(basedir, relpath)
        data = cls._from_file(filename)
        return ACO(time_stamp, fs, data, True, basedir=basedir)

    @classmethod
    def _ACO_to_int(cls, databytes, nbits):
        """
        Convert the block of bytes to an array of int32.

        We need to use int32 because there can be 17 bits.
        """
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
    def _params_from_filename(cls, filename):
        # 2016-02-15--05.00.HYD24BBpk
        name = osp.basename(filename)
        dts, encs = name.rsplit('.', 1)
        time_stamp = datetime.strptime(dts, cls.time_code)

        fs = int(re.findall('\d+', encs).pop()) * 1000
        return time_stamp, fs

    @classmethod
    def _from_file(cls, filename):
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


class _DatetimeACOLoader(_ACOLoader):
    res = timedelta(minutes=5)

    @classmethod
    def __floor_dt(cls, dt):
        src = timedelta(hours=dt.hour, minutes=dt.minute, seconds=dt.second)
        offset = src.total_seconds() % cls.res.total_seconds()
        return dt - timedelta(seconds=offset)

    @classmethod
    def _filename_from_date(cls, index_datetime):
        dts = datetime.strftime(index_datetime, cls.time_code)
        encs = 'HYD24BBpk'
        return '.'.join([dts, encs])

    @classmethod
    def _path_from_date(cls, index_datetime):
        info = [index_datetime.year, index_datetime.month, index_datetime.day]
        dirname = osp.join(*map(lambda i: str(i).zfill(2), info))
        basename = cls._filename_from_date(index_datetime)
        return osp.join(dirname, basename)

    @classmethod
    def load_ACO_from_datetime(
        cls,
        basedir, index_datetime,
        full=False, durration=timedelta(minutes=6)
    ):
        floor_datetime = cls.__floor_dt(index_datetime)
        aco = cls.load_ACO_from_file(
            basedir, cls._path_from_date(floor_datetime)
        )
        if full:
            return aco

        start = index_datetime - floor_datetime
        end = start + durration
        if end is not None:
            result = [aco]
            local_end = end
            while result[-1].end_datetime < result[-1].date_offset(local_end):
                _ = cls.load_ACO_from_datetime(
                    basedir,
                    result[-1].end_datetime + timedelta(minutes=1),
                    full=True)
                local_end = local_end - _._durration
                result.append(_)
            aco = reduce(ACO.__matmul__, result)

        return aco[start:end]


class ACOio:
    def __init__(self, basedir):
        self.basedir = basedir

    def load(self, target):
        if isinstance(target, str):
            return _ACOLoader.load_ACO_from_file(self.basedir, target)
        elif isinstance(target, datetime):
            return _DatetimeACOLoader.\
                load_ACO_from_datetime(self.basedir, target)
        else:
            raise TypeError


class ACO(Sound):
    def __init__(self, time_stamp, fs, data, raw=False, *, basedir):
        super().__init__(fs, data)
        self._time_stamp = time_stamp
        self.basedir = basedir
        self.raw = raw

    def copy(self):
        return ACO(
            self._time_stamp,
            self._fs,
            self._data.copy(),
            self.raw,
            basedir=self.basedir
        )

    @memoized_property
    def end_datetime(self):
        return self.date_offset(self._durration)

    def date_offset(self, durration):
        return self._time_stamp + durration

    def _date_difference(self, d):
        return self.durration_to_index(d - self._time_stamp)

    def __getitem__(self, slice_):
        result = self.copy()
        start = slice_.start
        timestamp = self._time_stamp + (
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

        ordered = sorted(ordered, key=attrgetter('_time_stamp'))
        durration = ordered[-1].end_datetime - ordered[0]._time_stamp

        space = ordered[0].durration_to_index(durration)

        data = np.full(space, np.NAN)

        idx = ~np.isnan(ordered[0]._data)
        data[:len(ordered[0]._data)][idx] = ordered[0]._data[idx]

        durration = ordered[-1]._time_stamp - ordered[0]._time_stamp
        start = ordered[0].durration_to_index(durration)

        idx = ~np.isnan(ordered[-1]._data)
        overlap_count = np.sum(~np.isnan(data[start:][idx]))

        data[start:][idx] = ordered[-1]._data[idx]

        if overlap_count > 0:
            warnings.warn(f'Overlaps {overlap_count} samples', UserWarning)

        result = self.__class__(
            ordered[0]._time_stamp,
            ordered[0]._fs,
            data,
            ordered[0].raw,
            basedir=self.basedir
        )
        return result
