from collections import namedtuple
from datetime import datetime, timedelta

import io

import numpy as np

import scipy.signal as signal
from scipy.io.wavfile import write as wavwrite
import scipy

# from python_speech_features import logfbank

from pyemd import emd as EMD
from memoized_property import memoized_property


PlotInfo = namedtuple('PlotInfo', ['data', 'xaxis', 'interval', 'shift'])


class AboveSoundBuonds(Exception):
    pass


class BelowSoundBounds(Exception):
    pass


class Sound:
    BULLSHITWAVNUMBER = 24000

    def __init__(self, fs, data):
        self._fs = fs
        self._data = data.astype(np.float64)

        self._opstack = []

    def random_sample(self, durration):
        t = np.random.uniform(0, (self._durration - durration).total_seconds())
        start = timedelta(seconds=t)
        return self[start:start+durration]

    def _getitem__indicies(self, slice_):
        i, j = slice_.start, slice_.stop

        new_start \
            = timedelta(0) if i is None else i

        new_end \
            = self._durration if j is None else j

        # if new_start.total_seconds() < 0:
        #     raise BelowSoundBounds
        #
        # if new_end.total_seconds() > self._durration.total_seconds():
        #     raise AboveSoundBuonds

        idx, jdx = self.durration_to_index(new_start), self.durration_to_index(new_end)
        return idx, jdx

    def __getitem__(self, slice_):
        idx, jdx = self._getitem__indicies(slice_)

        return Sound(
            self._fs,
            self._data[idx:jdx]
        )

    def chunk(self, durration, step):
        start = timedelta(seconds=0)
        stop = start + durration

        while stop.total_seconds() < self._durration.total_seconds():
            yield self[start:stop]
            start += step
            stop += step

    @property
    def header(self):
        return 'audio/x-wav'

    def get_wav(self):
        result = self.resample_fs(self.BULLSHITWAVNUMBER)
        data = result.normdata(dtype=np.int16)

        bytes_io = io.BytesIO()
        wavwrite(bytes_io, result._fs, data)

        return bytes_io

    @memoized_property
    def _durration(self):
        return timedelta(seconds=float((self._data.size / self._fs)))

    @memoized_property
    def _max_value(self):
        return np.max(np.abs(self._data))

    # @memoized_property
    def normdata(self, dtype=np.int32):
        data = self._data.copy().astype(np.float64)
        max_value = self._max_value
        data = ((data/max_value) * np.iinfo(dtype).max).astype(dtype)
        return data

    def _resample_fs(self, fs):
        fs_ratio = fs/self._fs
        print(fs_ratio)
        print(self._data)
        data = signal.resample(self._data, int(np.round(len(self)*fs_ratio)))
        return data

    def _highpass(self, data, BUTTER_ORDER, sampling_rate, cut_off):
        Wn = float(cut_off) / (float(sampling_rate) / 2.0)
        b, a = signal.butter(BUTTER_ORDER, Wn, 'high')
        return signal.filtfilt(b, a, data)

    def _lowpass(self, data, BUTTER_ORDER, sampling_rate, cut_off):
        Wn = float(cut_off) / (float(sampling_rate) / 2.0)
        b, a = signal.butter(BUTTER_ORDER, Wn, btype='low')
        return signal.filtfilt(b, a, data)

    def _pre_emphasis(self, data, pre_emphasis):
        return np.append(data[0], data[1:] - pre_emphasis * data[:-1])

    @memoized_property
    def _emd(self):
        emd = EMD()
        return emd(self._data)

    @classmethod
    def _to_frame_count(cls, fs, seconds):
        return int(np.round(seconds * fs))

    def to_frame_count(self, seconds):
        return self._to_frame_count(self._fs, seconds)

    def stft(self):
        return signal.stft(self._data, self._fs)

    def Listen(self, data=None):
        if data is None:
            data = self._data.copy()

        # bug in IPython.Audio, only handles common fs
        fs = self.BULLSHITWAVNUMBER
        data = self._resample_fs(fs)

        from IPython.display import Audio
        return Audio(data=data, rate=fs)

    def spectrogram(
        self, frame_duration=.08, frame_shift=.02, wtype='hanning'
    ):
        unit = self._Frame(frame_duration, frame_shift)
        mat = unit.data * signal.get_window(wtype, unit.data.shape[1])
        N = 2 ** int(np.ceil(np.log2(mat.shape[0])))
        return unit._replace(data=np.fft.rfft(mat, n=N))

    def power(self, frame_duration=.08, frame_shift=.02, wtype='boxcar'):
        num_overlap = self.to_frame_count(frame_duration - frame_shift)
        frame_size = self.to_frame_count(frame_duration)
        window = signal.get_window(wtype, frame_size)

        _, power = signal.welch(
            self._data,
            window=window,
            return_onesided=False,
            scaling='spectrum',
            noverlap=num_overlap
        )
        return power * window.sum()**2

    def _spectral_subtraction(
        self, other, frame_duration=.08, frame_shift=.02, wtype='boxcar'
    ):
        Frames = self._Frame(frame_duration, frame_shift).data
        power = other.power(frame_duration, frame_shift, wtype)
        window = signal.get_window(wtype, self.to_frame_count(frame_duration))

        spectrum = np.fft.fft(Frames * window)
        amplitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        # tune parameters
        α, β = 5.0, .02
        _ = (amplitude ** 2.0)
        __ = (power * α)
        _ = _ - __
        __ = amplitude ** 2
        __ = β * __
        _ = np.maximum(_, __)
        _ = np.sqrt(_)
        __ = phase * 1j
        __ = np.exp(__)
        _ = _ * __

        return _

    @classmethod
    def _overlap_add(cls, frames, shift, norm=True):
        count, size = frames.shape
        assert(shift < size)
        store = np.full((count, (size + (shift * (count - 1)))), np.NAN)
        for i in range(count):
            store[i][shift*i:shift*i+size] = frames[i]
        out = np.nansum(store, axis=0)
        if norm:
            out = out/np.sum(~np.isnan(store), axis=0)
        return out

    def logspectrogram(
        self, frame_duration=.08, frame_shift=.02, wtype='hanning'
    ):
        unit = self.spectrogram(frame_duration, frame_shift, wtype)
        return unit._replace(data=(20 * np.log10(np.abs(unit.data))))

    def autocorr(self):
        x = self._data
        n = len(x)
        return np.correlate(x, x, mode='full')[n - 1:]

    def periodogram(self):
        return signal.periodogram(self._data, fs=self._fs)

    def cepstrum(self, frame_duration=.08, frame_shift=.02, wtype='hanning'):
        unit = self.spectrogram(frame_duration, frame_shift, wtype)
        return unit._replace(
            data=(np.fft.irfft(np.log(np.abs(unit.data))).real)
        )

    def View(self, itype=None, **kwargs):
        if itype is None:
            unit = self._data
        elif hasattr(self, itype):
            attr = getattr(self, itype)
            unit = attr(**kwargs) if callable(attr) else attr
        else:
            raise "Fuck You"

        from matplotlib import pyplot as plt

        fig, ax = plt.subplots()
        name = 'wave' if itype is None else itype
        _ = plt.title(f'{name}')  # @ {self._time_stamp}')

        if isinstance(unit, PlotInfo):
            '''
            ['data', 'xaxis', 'yaxis'])
            _ = plt.plot(unit.data.T.real)
            '''

            _ = plt.imshow(X=unit.data.T.real, interpolation=None)
            _ = plt.yticks([])
            _ = plt.ylabel(
                f'{unit.interval:.3f} interval, {unit.shift:.3f} '
                'shift, {self._fs} f/s'
            )

        elif len(unit.shape) == 1:
            _ = plt.plot(unit)
        elif len(unit.shape) == 2:
            _ = plt.imshow(X=unit.T.real, interpolation=None)
        else:
            raise "DUM DUM DUM"

    def __len__(self):
        return len(self._data)

    def _Frame(self, frame_duration=.08, frame_shift=.02):
        n = self.to_frame_count(frame_duration)
        s = self.to_frame_count(frame_shift)

        total_frames = (len(self._data) - n) // s + 1
        zero = self._time_stamp if hasattr(self, '_time_stamp') \
            else datetime(1, 1, 1)
        time = (zero + (timedelta(seconds=frame_shift) * i)
                for i in range(total_frames))

        # dom = np.arange(total_frames) * s + n // 2
        mat = np.empty((total_frames, n))
        mat[:, :] = np.NAN

        start = 0
        for i in range(total_frames):
            idx = slice(start, (start+n))
            mat[i, :] = self._data[idx]
            start += s
        return PlotInfo(mat, time, frame_duration, frame_shift)

    def durration_to_index(self, t):
        return int(t.total_seconds() * self._fs)

    def _subtract_data(
        self, other, frame_duration=.08, frame_shift=.02, wtype='boxcar'
    ):
        assert(self._fs == other._fs)
        new_spectrum = self._spectral_subtraction(
            other, frame_duration, frame_shift, wtype
        )
        frames = np.fft.ifft(new_spectrum).real
        data = self._overlap_add(frames, self.to_frame_count(frame_shift))
        return data

    def subtract(
        self, other, frame_duration=.08, frame_shift=.02, wtype='boxcar'
    ):
        data = self._subtract_data(other, frame_duration, frame_shift, wtype)
        return Sound(
            self._fs,
            data
        )

    def save(self, label, store='train.csv'):
        data = self._data.copy()
        data.flags.writeable = False
        filename = f'{np.abs(hash(data.tobytes()))}.wav'
        scipy.io.wavfile.write(filename, self._fs, self._data.astype(np.int16))
        with open(store, mode='a') as fd:
            print(filename, label, sep=',', file=fd)
