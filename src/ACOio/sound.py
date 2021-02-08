from collections import namedtuple
from datetime import datetime, timedelta
import warnings

import io

import numpy as np

import scipy.signal as signal
from scipy.io.wavfile import write as wavwrite
import scipy

# from python_speech_features import logfbank

from memoized_property import memoized_property


PlotInfo = namedtuple('PlotInfo', ['data', 'xaxis', 'interval', 'shift'])


class NoViewMethodError(Exception):
    pass


class UnsupportedViewDimmensions(Exception):
    pass


class Sound:
    BULLSHITWAVNUMBER = 24000

    def __init__(self, fs, data):
        self._fs = fs
        self._data = data.astype(np.float64)

    def copy(self):
        return Sound(self._fs, self._data.copy())

    @classmethod
    def _resample_fs(cls, data, new_fs, old_fs):
        fs_ratio = new_fs/old_fs
        new_length = int(np.round(len(data) * fs_ratio))
        return signal.resample(data, new_length)

    def squash_nan(self):
        result = self.copy()
        idx = ~np.isnan(result._data)
        result._data = result._data[idx]
        return result

    def resample_fs(self, fs):
        ''' returns a track resampled to a specific frames per second '''
        result = self.copy()
        result._data = self._resample_fs(self._data, fs, self._fs)
        result._fs = fs
        return result

    def resample(self, n):
        ''' returns a track resampled to a specific number of data points '''
        result = self.copy()
        if len(self) != n:
            fs_ratio = n/len(self._data)
            warnings.warn(
                f'Only {fs_ratio:.3f} of signal represented',
                UserWarning
            )
            result._data = signal.resample(self._data, n)
            result._fs = int(np.round(self._fs * fs_ratio))
        return result

    def random_sample(self, durration):
        t = np.random.uniform(0, (self._durration - durration).total_seconds())
        start = timedelta(seconds=t)
        return self[start:start+durration]

    def chunk(self, durration, step):
        ''' returns generator to step over track for durration by step '''
        start = timedelta(seconds=0)
        stop = start + durration

        while stop.total_seconds() < self._durration.total_seconds():
            yield self[start:stop]
            start += step
            stop += step

    @classmethod
    def _pre_emphasis(cls, data, pre_emphasis):
        return np.append(data[0], data[1:] - pre_emphasis * data[:-1])

    def _getitem__indicies(self, slice_):
        i, j = slice_.start, slice_.stop

        new_start \
            = timedelta(0) if i is None else i
        new_end \
            = self._durration if j is None else j

        idx, jdx = \
            self.durration_to_index(new_start), \
            self.durration_to_index(new_end)
        return idx, jdx

    def __getitem__(self, slice_):
        idx, jdx = self._getitem__indicies(slice_)
        result = self.copy()
        result._data = result._data[idx:jdx]
        return result

    @memoized_property
    def _durration(self):
        return timedelta(seconds=float((self._data.size / self._fs)))

    @classmethod
    def _to_frame_count(cls, fs, seconds):
        return int(np.round(seconds * fs))

    def to_frame_count(self, seconds):
        ''' given a seconds count, return the index offsed count for track '''
        return self._to_frame_count(self._fs, seconds)

    def __len__(self):
        return len(self._data)

    def durration_to_index(self, t):
        ''' returns the idx offset reached when stepping `t` seconds '''
        return int(t.total_seconds() * self._fs)

    @memoized_property
    def _max_value(self):
        return np.max(np.abs(self._data))

    def normdata(self, dtype=np.int32):
        ''' safe normalization of `._data` to the specified bit precision '''
        data = self._data.copy().astype(np.float64)
        max_value = self._max_value
        data = ((data/max_value) * (np.iinfo(dtype).max - 1)).astype(dtype)
        return data

    @classmethod
    def _lowpass(cls, data, BUTTER_ORDER, sampling_rate, cut_off):
        Wn = float(cut_off) / (float(sampling_rate) / 2.0)
        b, a = signal.butter(BUTTER_ORDER, Wn, btype='low')
        return signal.filtfilt(b, a, data)

    def lowpass(self, BUTTER_ORDER=6, cut_off=3000.0):
        ''' returns track after application of low-pass buttworth filter '''
        result = self.copy()
        result._data = self._lowpass(
            self._data,
            BUTTER_ORDER=BUTTER_ORDER,
            sampling_rate=self._fs,
            cut_off=cut_off)
        return result

    @classmethod
    def _highpass(cls, data, BUTTER_ORDER, sampling_rate, cut_off):
        Wn = float(cut_off) / (float(sampling_rate) / 2.0)
        b, a = signal.butter(BUTTER_ORDER, Wn, 'high')
        return signal.filtfilt(b, a, data)

    def highpass(self, BUTTER_ORDER=6, cut_off=30.0):
        ''' returns track after application of high-pass buttworth filter '''
        result = self.copy()
        result._data = self._highpass(
            self._data,
            BUTTER_ORDER=BUTTER_ORDER,
            sampling_rate=self._fs,
            cut_off=cut_off)
        return result

    def stft(self):
        ''' short term fourier transform, as implemented by `signal.stft` '''
        return signal.stft(self._data, self._fs)

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

    # def periodogram(self):
    #     return signal.periodogram(self._data, fs=self._fs)

    def autocorr(self, mode='full'):
        x = self._data
        n = len(x)
        return np.correlate(x, x, mode=mode)[n - 1:]

    def logspectrogram(
        self, frame_duration=.08, frame_shift=.02, wtype='hanning'
    ):
        unit = self.spectrogram(frame_duration, frame_shift, wtype)
        return unit._replace(data=(20 * np.log10(np.abs(unit.data))))

    def cepstrum(self, frame_duration=.08, frame_shift=.02, wtype='hanning'):
        unit = self.spectrogram(frame_duration, frame_shift, wtype)
        return unit._replace(
            data=(np.fft.irfft(np.log(np.abs(unit.data))).real)
        )

    def spectrogram(
        self, frame_duration=.08, frame_shift=.02, wtype='hanning'
    ):
        unit = self._Frame(frame_duration, frame_shift)
        mat = unit.data * signal.get_window(wtype, unit.data.shape[1])
        N = 2 ** int(np.ceil(np.log2(mat.shape[0])))
        return unit._replace(data=np.fft.rfft(mat, n=N))

    def _Frame(self, frame_duration=.08, frame_shift=.02):
        '''
        returns a sliding frame view, turning the track into a 2d array,
        specified by frame_duration and frame_shift, usually precedes
        application of windowing, then onto iltering
        '''
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

    def _spectral_subtraction(
        self, other, α, β,
        frame_duration=.08, frame_shift=.02, wtype='boxcar'
    ):
        Frames = self._Frame(frame_duration, frame_shift).data
        power = other.power(frame_duration, frame_shift, wtype)
        window = signal.get_window(wtype, self.to_frame_count(frame_duration))

        spectrum = np.fft.fft(Frames * window)
        amplitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        # theres lots of math in these parts
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

    def _subtract_data(
        self, other, α=5.0, β=.02,
        frame_duration=.08, frame_shift=.02, wtype='boxcar'
    ):
        assert(self._fs == other._fs)
        new_spectrum = self._spectral_subtraction(
            other, α, β, frame_duration, frame_shift, wtype
        )
        frames = np.fft.ifft(new_spectrum).real
        data = self._overlap_add(frames, self.to_frame_count(frame_shift))
        return data

    def subtract(
        self, other, α=5.0, β=.02,
        frame_duration=.08, frame_shift=.02, wtype='boxcar'
    ):
        ''' returns new track after application of spectral subtraction '''
        result = self.copy()
        result._data = self._subtract_data(
            other, α, β, frame_duration, frame_shift, wtype
        )
        return result

    def Listen(self, data=None):
        '''
        creates a jypyter compliant audio component, always resampled to
        `self.BULLSHITWAVNUMBER` frames per second. This is required in order
        to play the track in a browser. For most accurate listening, consider
        saving out the content and using `sox` audio player.
        '''
        if data is None:
            data = self._data.copy()

        # cannot resample values with nan
        idx = np.isnan(data)
        data[idx] = 0

        # bug in IPython.Audio, only handles common fs
        data = self._resample_fs(self._data, self.BULLSHITWAVNUMBER, self._fs)

        from IPython.display import Audio
        return Audio(data=data, rate=self.BULLSHITWAVNUMBER)

    def View(self, itype=None, **kwargs):
        '''
        Shortcut method to jupyter compliant viewing of track under supported
        methods. `itype` defines which visualization to provide.

        Supported methods are transforms that yield a 1d or 2d numpy array.
            None => wav files
            'spectrogram', 'logspectrogram', 'power', 'pariodogram',
            'cepstrum', ...

        `**kwargs` specify the parameters to the `itype` method. See associated
            method signatures. Sane defaults are selected.
        '''

        if itype is None:
            unit = self._data
        elif hasattr(self, itype):
            attr = getattr(self, itype)
            unit = attr(**kwargs) if callable(attr) else attr
        else:
            raise NoViewMethodError

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
                f'shift, {self._fs} f/s'
            )

        elif len(unit.shape) == 1:
            _ = plt.plot(unit)
        elif len(unit.shape) == 2:
            _ = plt.imshow(X=unit.T.real, interpolation=None)
        else:
            raise UnsupportedViewDimmensions

    @property
    def header(self):
        return 'audio/x-wav'

    def get_wav(self, *, resample=True):
        '''Retrieves io.BytesIO() packed with `.wav` contents'''
        result = self.resample_fs(self.BULLSHITWAVNUMBER) if resample \
            else self.copy()
        data = result.normdata(dtype=np.int16)

        bytes_io = io.BytesIO()
        wavwrite(bytes_io, result._fs, data)

        return bytes_io

    def save(self, label, note="", store='saved.csv'):
        data = self._data.copy()
        data.flags.writeable = False
        filename = f'{np.abs(hash(data.tobytes()))}.wav'
        scipy.io.wavfile.write(filename, self._fs, self.normdata(np.int16))
        with open(store, mode='a') as fd:
            _ = note.replace(',', ':comma:')
            print(filename, label, _, sep=',', file=fd)
