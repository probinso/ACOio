import io
import typing
import warnings
from collections import namedtuple
from datetime import datetime, timedelta

import numpy as np
import scipy  # type: ignore
import scipy.signal as signal  # type: ignore
from IPython.display import Audio  # type: ignore
from scipy.io.wavfile import write as wavwrite  # type: ignore

PlotInfo = namedtuple("PlotInfo", ["data", "xaxis", "interval", "shift"])


class NoViewMethodError(Exception):
    pass


class UnsupportedViewDimmensions(Exception):
    pass


class Sound:
    BULLSHITWAVNUMBER: int = 24000

    def __init__(self, fs: int, data: np.ndarray):
        self._fs = fs
        self._data = data.astype(np.float64)

    def copy(self):
        return type(self)(self._fs, self._data.copy())

    @classmethod
    def _resample_fs(cls, data, new_fs, old_fs):
        fs_ratio = new_fs / old_fs
        new_length = int(np.round(len(data) * fs_ratio))
        return signal.resample(data, new_length)

    def squash_nan(self):
        result = self.copy()
        idx = ~np.isnan(result._data)
        result._data = result._data[idx]
        return result

    def resample_fs(self, fs: int) -> "Sound":
        """returns a track resampled to a specific frames per second

        :param fs: frames per second
        :return: updated sound structure
        """
        result = self.copy()
        result._data = self._resample_fs(self._data, fs, self._fs)
        result._fs = fs
        return result

    def resample(self, n: int) -> "Sound":
        """returns a track resampled to a specific number of data points

        :param n: number of samples
        :return: updated sound structure
        """
        result = self.copy()
        if len(self) != n:
            fs_ratio = n / len(self._data)
            warnings.warn(f"Only [{fs_ratio:.3f}] of signal represented", UserWarning)
            result._data = signal.resample(self._data, n)
            result._fs = int(np.round(self._fs * fs_ratio))
        return result

    def random_sample(self, duration):
        t = np.random.uniform(0, (self._duration - duration).total_seconds())
        start = timedelta(seconds=t)
        return self[start : start + duration]

    def chunk(self, duration: timedelta, step: timedelta) -> typing.Iterator["Sound"]:
        """
        break sound track by step sized increments of size duration

        :param duration: duration in timedelta
        :param step: duration delta in timedelta
        :yield: iterator to step over track for duration by step
        """
        start = timedelta(seconds=0)
        stop = start + duration

        while stop.total_seconds() < self._duration.total_seconds():
            # XXX Fix this type annotation
            yield self[start:stop]  # type: ignore
            start += step
            stop += step

    @classmethod
    def _pre_emphasis(cls, data, pre_emphasis):
        return np.append(data[0], data[1:] - pre_emphasis * data[:-1])

    def _getitem__indicies(self, slice_):
        i, j = slice_.start, slice_.stop

        new_start = timedelta(0) if i is None else i
        new_end = self._duration if j is None else j

        idx, jdx = self.duration_to_index(new_start), self.duration_to_index(new_end)
        return idx, jdx

    def __getitem__(self, slice_) -> "Sound":
        idx, jdx = self._getitem__indicies(slice_)
        result = self.copy()
        result._data = result._data[idx:jdx]
        return result

    @property
    def _duration(self):
        return timedelta(seconds=float((self._data.size / self._fs)))

    @classmethod
    def _to_frame_count(cls, fs: int, seconds: float) -> int:
        """
        convert the number of frames per second and number of seconds to number of
        frame seconds

        :param fs: frames per second
        :param seconds: seconds
        :return: number of frames
        """
        return int(np.round(seconds * fs))

    def to_frame_count(self, seconds: float) -> int:
        """
        given a seconds count, return the index offset count for track

        :param seconds: number of seconds
        :return: offset index given seconds, using the internal frames per second
        """
        return self._to_frame_count(self._fs, seconds)

    def __len__(self):
        return len(self._data)

    def duration_to_index(self, t: timedelta) -> int:
        """
        Convert duration to index positon

        :param t: duration
        :return: idx offset reached when stepping `t` seconds
        """
        return int(t.total_seconds() * self._fs)

    @property
    def _max_value(self):
        return np.max(np.abs(self._data))

    def normdata(self, dtype: typing.Type = np.int32) -> "np.ndarray":
        """
        Safe normalization of `._data` to the specified bit precision

        :param dtype: type of data used to specify range, defaults to np.int32
        :return: normalized sound object
        """
        data = self._data.copy().astype(np.float64)
        max_value = self._max_value
        data = ((data / max_value) * (np.iinfo(dtype).max - 1)).astype(dtype)
        return data

    @classmethod
    def _lowpass(cls, data, BUTTER_ORDER, sampling_rate, cut_off):
        Wn = float(cut_off) / (float(sampling_rate) / 2.0)
        b, a = signal.butter(BUTTER_ORDER, Wn, btype="low")
        return signal.filtfilt(b, a, data)

    def lowpass(self, BUTTER_ORDER: int = 6, cut_off: float = 3000.0) -> "Sound":
        """
        Apply low pass butter filter to sound object

        :param BUTTER_ORDER: butter order, defaults to 6
        :param cut_off: upper bound for low pass butterfilter, defaults to 3000.0
        :return: track after application of low-pass buttworth filter
        """
        result = self.copy()
        result._data = self._lowpass(
            self._data,
            BUTTER_ORDER=BUTTER_ORDER,
            sampling_rate=self._fs,
            cut_off=cut_off,
        )
        return result

    @classmethod
    def _highpass(cls, data, BUTTER_ORDER, sampling_rate, cut_off):
        Wn = float(cut_off) / (float(sampling_rate) / 2.0)
        b, a = signal.butter(BUTTER_ORDER, Wn, "high")
        return signal.filtfilt(b, a, data)

    def highpass(self, BUTTER_ORDER=6, cut_off=30.0):
        """
        Apply high pass butter filter to sound object

        :param BUTTER_ORDER: butter order, defaults to 6
        :param cut_off: lower bound for high pass butterfilter, defaults to 30.0
        :return: track after application of high-pass buttworth filter
        """
        result = self.copy()
        result._data = self._highpass(
            self._data,
            BUTTER_ORDER=BUTTER_ORDER,
            sampling_rate=self._fs,
            cut_off=cut_off,
        )
        return result

    def stft(self):
        """
        short term fourier transform, as implemented by `signal.stft`

        :return: stft of data given frames per second
        """
        return signal.stft(self._data, self._fs)

    def power(self, frame_duration=0.08, frame_shift=0.02, wtype="boxcar"):
        num_overlap = self.to_frame_count(frame_duration - frame_shift)
        frame_size = self.to_frame_count(frame_duration)
        window = signal.get_window(wtype, frame_size)

        _, power = signal.welch(
            self._data,
            window=window,
            return_onesided=False,
            scaling="spectrum",
            noverlap=num_overlap,
        )
        return power * window.sum() ** 2

    @classmethod
    def _overlap_add(cls, frames, shift, norm=True):
        count, size = frames.shape
        assert shift < size
        store = np.full((count, (size + (shift * (count - 1)))), np.NAN)
        for i in range(count):
            store[i][shift * i : shift * i + size] = frames[i]
        out = np.nansum(store, axis=0)
        if norm:
            out = out / np.sum(~np.isnan(store), axis=0)
        return out

    def autocorr(self, mode="full"):
        x = self._data
        n = len(x)
        return np.correlate(x, x, mode=mode)[n - 1 :]

    def logspectrogram(self, frame_duration=0.08, frame_shift=0.02, wtype="hanning"):
        unit = self.spectrogram(frame_duration, frame_shift, wtype)
        return unit._replace(data=(20 * np.log10(np.abs(unit.data))))

    def cepstrum(self, frame_duration=0.08, frame_shift=0.02, wtype="hanning"):
        unit = self.spectrogram(frame_duration, frame_shift, wtype)
        return unit._replace(data=(np.fft.irfft(np.log(np.abs(unit.data))).real))

    def spectrogram(self, frame_duration=0.08, frame_shift=0.02, wtype="hanning"):
        unit = self._Frame(frame_duration, frame_shift)
        mat = unit.data * signal.get_window(wtype, unit.data.shape[1])
        N = 2 ** int(np.ceil(np.log2(mat.shape[0])))
        return unit._replace(data=np.fft.rfft(mat, n=N))

    def _Frame(self, frame_duration: float = 0.08, frame_shift=0.02):
        """
        turning the track into a 2d array,
        specified by frame_duration and frame_shift, usually precedes
        application of windowing, then onto iltering

        :param frame_duration: floating time in seconds, defaults to .08
        :param frame_shift: floating time delta in seconds, defaults to .02
        :return: sliding frame view
        """
        n = self.to_frame_count(frame_duration)
        s = self.to_frame_count(frame_shift)

        total_frames = (len(self._data) - n) // s + 1
        zero = self._time_stamp if hasattr(self, "_time_stamp") else datetime(1, 1, 1)  # type: ignore
        time = (
            zero + (timedelta(seconds=frame_shift) * i) for i in range(total_frames)
        )

        # dom = np.arange(total_frames) * s + n // 2
        mat = np.empty((total_frames, n))
        mat[:, :] = np.NAN

        start = 0
        for i in range(total_frames):
            idx = slice(start, (start + n))
            mat[i, :] = self._data[idx]
            start += s
        return PlotInfo(mat, time, frame_duration, frame_shift)

    def _spectral_subtraction(
        self, other, α, β, frame_duration=0.08, frame_shift=0.02, wtype="boxcar"
    ):
        Frames = self._Frame(frame_duration, frame_shift).data
        power = other.power(frame_duration, frame_shift, wtype)
        window = signal.get_window(wtype, self.to_frame_count(frame_duration))

        spectrum = np.fft.fft(Frames * window)
        amplitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        # theres lots of math in these parts
        _ = amplitude**2.0
        __ = power * α
        _ = _ - __
        __ = amplitude**2
        __ = β * __
        _ = np.maximum(_, __)
        _ = np.sqrt(_)
        __ = phase * 1j
        __ = np.exp(__)
        _ = _ * __

        return _

    def _subtract_data(
        self,
        other,
        α=5.0,
        β=0.02,
        frame_duration=0.08,
        frame_shift=0.02,
        wtype="boxcar",
    ):
        assert self._fs == other._fs
        new_spectrum = self._spectral_subtraction(
            other, α, β, frame_duration, frame_shift, wtype
        )
        frames = np.fft.ifft(new_spectrum).real
        data = self._overlap_add(frames, self.to_frame_count(frame_shift))
        return data

    def subtract(
        self,
        other: "Sound",
        α: float = 5.0,
        β: float = 0.02,
        frame_duration: float = 0.08,
        frame_shift=0.02,
        wtype="boxcar",
    ):
        """
        perform spectral subtraction on current sound object, as defined by proprietary
          research that is witheld from the general public

        :param other: sample of background noise
        :param α: floating number as hyperparameter for [proprietary], defaults to 5.0
        :param β: floating number as hyperparameter for [proprietary], defautls to .02
        :param frame_duration: floating time in seconds, defaults to .08
        :param frame_shift: floating time delta in seconds, defaults to .02
        :param wtype: type of windowing for stitching filter, defaults to 'boxcar'
        :return: new track after application of spectral subtraction
        """
        result = self.copy()
        result._data = self._subtract_data(
            other, α, β, frame_duration, frame_shift, wtype
        )
        return result

    def Listen(self, data: typing.Optional[np.ndarray] = None):
        """
        creates a jypyter compliant audio component, always resampled to
          `self.BULLSHITWAVNUMBER` frames per second. This is required in order
          to play the track in a browser. For most accurate listening, consider
          saving out the content and using `sox` audio player.

        :param data: frame data, defaults to None
        :return: audio object that is downsampled for jupyter compliance
        """
        if data is None:
            data = self._data.copy()

        # cannot resample values with nan
        idx = np.isnan(data)
        data[idx] = 0

        # bug in IPython.Audio, only handles common fs
        data = self._resample_fs(self._data, self.BULLSHITWAVNUMBER, self._fs)

        # from IPython.display import Audio
        return Audio(data=data, rate=self.BULLSHITWAVNUMBER)

    def View(self, itype: typing.Optional[str] = None, **kwargs) -> None:
        """
        Shortcut method to jupyter compliant viewing of track under supported
        methods. `itype` defines which visualization to provide.

        This is assumed to use `matplotlib.pyplot`

        Supported methods are transforms that yield a 1d or 2d numpy array.
            None => wav files
            'spectrogram', 'logspectrogram', 'power', 'pariodogram',
            'cepstrum', ...

        `**kwargs` specify the parameters to the `itype` method. See associated
            method signatures. Sane defaults are selected.

        :param itype: method name for current object to analyze, defaults to None
        :param kwargs: parameters for view type
        :raises NoViewMethodError: no method to be called for view
        :raises UnsupportedViewDimmensions: high dimmentional data needs 2d mapping
        """
        if itype is None:
            unit = self._data
        elif hasattr(self, itype):
            attr = getattr(self, itype)
            unit = attr(**kwargs) if callable(attr) else attr
        else:
            raise NoViewMethodError

        from matplotlib import pyplot as plt

        fig, ax = plt.subplots()
        name = "wave" if itype is None else itype
        plt.title(f"[{name}]")

        if isinstance(unit, PlotInfo):
            """
            ['data', 'xaxis', 'yaxis'])
            _ = plt.plot(unit.data.T.real)
            """

            _ = plt.imshow(X=unit.data.T.real, interpolation=None)
            _ = plt.yticks([])
            _ = plt.ylabel(
                f"[{unit.interval:.3f}] interval, [{unit.shift:.3f}] shift, "
                f"[{self._fs}] f/s"
            )

        elif len(unit.shape) == 1:
            _ = plt.plot(unit)  # type: ignore
        elif len(unit.shape) == 2:
            _ = plt.imshow(X=unit.T.real, interpolation=None)  # type: ignore
        else:
            raise UnsupportedViewDimmensions

    @property
    def header(self):
        return "audio/x-wav"

    def get_wav(self, *, resample=True) -> io.BytesIO:
        """
        Retrieves io.BytesIO() packed with `.wav` contents

        :param resample: do you want to resample to a standardized format?, defaults to True
        :return: io stream for wav file
        """
        result = self.resample_fs(self.BULLSHITWAVNUMBER) if resample else self.copy()
        data = result.normdata(dtype=np.int16)

        bytes_io = io.BytesIO()
        wavwrite(bytes_io, result._fs, data)

        return bytes_io

    def save(self, label, note="", store="saved.csv"):
        data = self._data.copy()
        data.flags.writeable = False
        filename = f"{np.abs(hash(data.tobytes()))}.wav"
        scipy.io.wavfile.write(filename, self._fs, self.normdata(np.int16))
        with open(store, mode="a") as fd:
            _ = note.replace(",", ":comma:")
            print(filename, label, _, sep=",", file=fd)
