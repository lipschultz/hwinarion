import io
from array import array
from typing import Iterable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from pydub import AudioSegment
from pydub.playback import play

TimeType = Union[float, int]


class AudioSample:
    def __init__(self, data: AudioSegment):
        self.data = data

    def __eq__(self, other: "AudioSample") -> bool:
        """
        Returns True if other object is an ``AudioSample`` object and they have the same data.
        """
        return isinstance(other, AudioSample) and self.data == other.data

    def __len__(self):
        """
        The number of frames in the audio sample.
        """
        return self.n_frames

    @property
    def n_frames(self) -> int:
        """
        The number of frames in the audio sample.
        """
        return int(self.data.frame_count())

    @property
    def n_seconds(self) -> float:
        """
        The number of seconds in the audio sample.
        """
        return self.data.duration_seconds

    def slice_time(self, start_stop: TimeType, stop: Optional[TimeType] = None) -> "AudioSample":
        """
        Get a continuous section of audio defined by ``start_stop`` and ``stop``.  The parameters follow Python slice
        parameters.  Both ``start_stop`` and ``stop`` are in seconds.  Negative values are not currently supported

        If ``stop`` is ``None`` (default), then ``start_stop`` defines the ending time, with the start time being the
        beginning of the audio sample.

        If ``stop`` is a number, then ``start_stop`` defines the start time.
        """
        # TODO: Support negative values for slices.  Confirm start_stop <= stop.
        if stop is None:
            start = 0
            stop = start_stop
        else:
            start = start_stop

        # pydub slices in milliseconds
        start_ms = start * 1000
        stop_ms = stop * 1000
        return AudioSample(self.data[start_ms:stop_ms])

    def slice_frame(self, start_stop: int, stop: Optional[int] = None) -> "AudioSample":
        """
        Get a continuous section of audio defined by ``start_stop`` and ``stop``.  The parameters follow Python slice
        parameters.  Both ``start_stop`` and ``stop`` are number of frames.  Negative values are not currently supported

        If ``stop`` is ``None`` (default), then ``start_stop`` defines the ending frame, with the start frame being the
        beginning of the audio sample.

        If ``stop`` is a number, then ``start_stop`` defines the start frame.
        """
        # TODO: Support negative values for slices.  Confirm start_stop <= stop.
        if stop is None:
            start = 0
            stop = start_stop
        else:
            start = start_stop

        # pydub slices in milliseconds
        return self.from_numpy_and_sample(self.to_numpy()[start:stop], self)

    def to_bytes(self) -> bytes:
        """
        Get the audio sample as bytes.
        """
        return self.data.raw_data

    def to_numpy(self) -> np.ndarray:
        """
        Get the audio sample as a numpy array.
        """
        return np.frombuffer(self.to_bytes(), dtype=f"int{self.bit_depth}")

    def to_array(self) -> array:
        """
        Get the audio sample as an array.
        """
        return self.data.get_array_of_samples()

    def seconds_to_frame(self, seconds: TimeType) -> int:
        """
        Convert time (in seconds) into number of frames.
        """
        return int(seconds * self.frame_rate)

    def frame_to_seconds(self, frame: int) -> float:
        """
        Convert number of frames into time (in seconds).
        """
        return frame / self.frame_rate

    @property
    def frame_rate(self) -> int:
        """
        Frames per second of the audio sample.
        """
        return self.data.frame_rate

    @property
    def n_channels(self) -> int:
        """
        Number of audio channels in the audio sample (1 for mono, 2 for stereo, ...).
        """
        return self.data.channels

    @property
    def bit_depth(self) -> int:
        """
        The number of bits dedicated to each sample.
        """
        return self.sample_width * 8

    @property
    def frame_width(self) -> int:
        """
        The width of an individual frame, which will consist of
        ``self.n_channels`` samples.  When there's just one channel,
        then frame_width is the same as sample_width, but with two
        channels ``frame_width`` will be double ``sample_width``.
        """
        return self.data.frame_width

    @property
    def sample_width(self) -> int:
        """
        The width of a sample from just one channel.
        """
        return self.data.sample_width

    @property
    def rms(self) -> int:
        """
        A measure of the loudness or energy in the audio.
        """
        return self.data.rms

    @property
    def max_possible_amplitude(self) -> float:
        """
        The maximum possible amplitude (based on the bit depth of each sample).
        """
        return (2**self.bit_depth) / 2  # The "/2" is because half of the size is for above 0, the other for below

    def convert(
        self,
        frame_rate: Optional[int] = None,
        sample_width: Optional[int] = None,
        n_channels: Optional[int] = None,
    ) -> "AudioSample":
        """
        Convert the audio sample into a new audio sample with a different frame rate, sample width, and/or number of
        channels.
        """
        new_data = self.data

        if frame_rate is not None:
            new_data = new_data.set_frame_rate(frame_rate)

        if sample_width is not None:
            new_data = new_data.set_sample_width(sample_width)

        if n_channels is not None:
            new_data = new_data.set_channels(n_channels)

        return AudioSample(new_data)

    def invert_phase(self) -> "AudioSample":
        """
        Invert the phase of the audio sample.
        """
        return AudioSample(self.data.invert_phase())

    def normalize(self) -> "AudioSample":
        return AudioSample(self.data.normalize(0))

    def append(self, audio_sample: "AudioSample", crossfade: TimeType = 0) -> "AudioSample":
        """
        Return a new ``AudioSample`` where ``audio_sample`` is appended onto the end of the current audio sample.

        If ``crossfade`` is not zero, then it represents the amount of overlap (in seconds) of the two audio sample.
        """
        return AudioSample(self.data.append(audio_sample.data, crossfade=crossfade * 1000))  # pydub works in ms

    def overlay(self, audio_sample: "AudioSample", offset: TimeType = 0) -> "AudioSample":
        """
        Return a new ``AudioSample`` where ``audio_sample`` and the current audio sample are overlaid on top of each
        other.

        ``offset`` represents the number of seconds to delay ``audio_sample`` in the final overlaid result.

        If ``audio_sample`` and the offset together are longer than the current sample, then ``audio_sample`` is
        truncated to fit within the duration of the current sample.
        """
        return AudioSample(self.data.overlay(audio_sample.data, position=int(offset * 1000)))  # pydub works in ms

    @classmethod
    def from_iterable(cls, audio_samples: Iterable["AudioSample"], crossfade: TimeType = 0) -> Optional["AudioSample"]:
        """
        Create an audio sample by concatenating ``audio_samples`` together.

        If ``crossfade`` is not zero, then it represents the amount of overlap (in seconds) of the two audio sample.
        """
        final_sample = None
        for sample in audio_samples:
            if final_sample is None:
                final_sample = sample
            else:
                final_sample = final_sample.append(sample, crossfade=crossfade)
        return final_sample

    @classmethod
    def generate_silence(cls, n_seconds: TimeType, frame_rate: int) -> "AudioSample":
        return AudioSample(AudioSegment.silent(duration=int(n_seconds * 1000), frame_rate=frame_rate))

    @classmethod
    def from_numpy(cls, data: np.ndarray, frame_rate: int) -> "AudioSample":
        return AudioSample(
            AudioSegment(
                data.tobytes(),
                frame_rate=frame_rate,
                sample_width=data.dtype.itemsize,
                channels=1 if len(data.shape) == 1 else data.shape[1],
            )
        )

    @classmethod
    def from_numpy_and_sample(cls, data: np.ndarray, source_sample: "AudioSample") -> "AudioSample":
        return cls.from_numpy(data, source_sample.frame_rate)

    def play(self, delta_gain_dB: float = None) -> None:
        """
        Play the audio sample.

        If ``delta_gain_dB`` is a number, then adjust the gain before playing.  This change is temporary, only for
        playing the audio sample.  If ``None`` (default), then don't adjust the gain.
        """
        data = self.data
        if delta_gain_dB is not None:
            data = data.apply_gain(delta_gain_dB)
        play(data)

    def export(self, filename, **kwargs) -> io.BufferedRandom:
        return self.data.export(filename, **kwargs)

    def remove_dc_offset(self, rolling: bool = False) -> "AudioSample":
        if not rolling:
            return AudioSample(self.data.remove_dc_offset())

        # This should work, but I lose half the audio when converting the numpy array back into an AudioSegment
        window_size = 15

        samples = self.to_numpy()
        view = sliding_window_view(samples, window_size)
        frame_averages = np.concatenate(
            (
                view.mean(axis=-1).astype(int),
                [int(samples[-(window_size - 1) :].mean())] * (window_size - 1),
            )
        )

        adjusted = samples - frame_averages
        return AudioSample.from_numpy_and_sample(adjusted, self)

    def plot_amplitude(
        self,
        *,
        title=None,
        axis=None,
        highlight_regions: Iterable[Tuple[TimeType, TimeType]] = (),
        normalize_amplitude: bool = False,
    ) -> None:
        """
        Plot the amplitude of the audio sample in a matplotlib graph.  The x-axis is time in seconds.

        ``title`` is the title put on the graph, or ``None`` (default) to have no title.

        ``axis`` is the axis to plot on.  If ``axis`` is ``None`` (default), then an axis will be created to plot on and
        the plot will be shown.  If ``axis`` is not ``None``, then the plot will not be shown (because it's assumed
        the caller will be plotting other items before showing).

        ``highlight_regions`` is an iterable of (start time, end time) tuples (time in seconds) to highlight in the
        plot.

        ``normalize_amplitude`` is a bool indicating whether to normalize the amplitude (when ``True``, default) or not
        (when ``False``) before plotting it.
        """
        time = np.linspace(0, self.n_seconds, num=len(self))

        if normalize_amplitude:
            normed_sample = self.normalize()
            samples = normed_sample.to_numpy() / normed_sample.max_possible_amplitude
        else:
            samples = self.to_numpy()

        show_plot = axis is None
        if axis is None:
            _, axis = plt.subplots()

        if title is not None:
            axis.set_title(title)
        axis.set_xlabel("Time (sec)")
        axis.set_ylabel("Amplitude")

        amplitude_line, *_ = axis.plot(time, samples)
        amplitude_color = amplitude_line.get_color()

        for region in highlight_regions:
            axis.fill_between(
                region,
                samples.min(),
                samples.max(),
                facecolor=amplitude_color,
                alpha=0.3,
            )

        if show_plot:
            plt.show()

    plot = plot_amplitude

    def plot_spectrogram(self, title=None, *, mode="psd", axis=None):
        """
        Plot the spectrogram of the audio sample in a matplotlib graph.

        ``title`` is the title put on the graph, or ``None`` (default) to have no title.

        ``mode`` is which spectrogram mode to pass to matplotlib.

        ``axis`` is the axis to plot on.  If ``axis`` is ``None`` (default), then an axis will be created to plot on and
        the plot will be shown.  If ``axis`` is not ``None``, then the plot will not be shown (because it's assumed
        the caller will be plotting other items before showing).
        """
        samples = self.to_numpy()

        show_plot = axis is None
        if axis is None:
            _, axis = plt.subplots()

        if title is not None:
            axis.set_title(title)
        axis.specgram(samples, mode=mode)

        if show_plot:
            plt.show()

    def plot_all(self):
        """
        Produce a plot with two sub-graphs, one for the amplitude and the other for the spectrogram.
        """
        fig, (ax_amplitude, ax_spectrogram) = plt.subplots(nrows=2)

        self.plot_amplitude(axis=ax_amplitude)
        self.plot_spectrogram(axis=ax_spectrogram)

        plt.show()


class BaseAudioSource:
    @property
    def frame_rate(self) -> int:
        raise NotImplementedError

    def seconds_to_frame(self, seconds: TimeType) -> int:
        """
        Convert time (in seconds) into number of frames.
        """
        return int(seconds * self.frame_rate)

    def read(self, n_frames: Optional[int]) -> AudioSample:
        """
        Retrieve up to ``n_frames`` of audio data from the audio source.  Depending on implementation, it may block
        until all frames are read or just return what has been retrieved.  Audio is returned in an ``AudioSample``
        object.
        """
        raise NotImplementedError

    def read_seconds(self, n_seconds: Optional[TimeType]) -> AudioSample:
        """
        Retrieve up to ``n_seconds`` of audio data from the audio source.  Depending on implementation, it may block
        until all frames are read or just return what has been retrieved.  Audio is returned in an ``AudioSample``
        object.
        """
        if n_seconds is not None:
            if not isinstance(n_seconds, TimeType) or n_seconds <= 0:
                raise TypeError(f"n_seconds must be None or a number greater than zero, got {n_seconds!r}")

        if n_seconds is None:
            n_frames = None
        else:
            n_frames = self.seconds_to_frame(n_seconds)
        return self.read(n_frames)
