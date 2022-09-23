import io
from array import array
from typing import Union, Optional, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from pydub.playback import play

TimeType = Union[float, int]


class BaseAudioSource:
    @property
    def frame_rate(self) -> int:
        raise NotImplementedError

    def seconds_to_frame(self, seconds: TimeType) -> int:
        return int(seconds * self.frame_rate)

    def read(self, n_frames: Optional[int]) -> 'AudioSample':
        raise NotImplementedError

    def read_seconds(self, n_seconds: Optional[TimeType]) -> 'AudioSample':
        assert n_seconds is None or (isinstance(n_seconds, (int, float)) and n_seconds > 0), f'n_seconds must be None or a number greater than zero, got {n_seconds!r}'

        if n_seconds is None:
            n_frames = None
        else:
            n_frames = self.seconds_to_frame(n_seconds)
        return self.read(n_frames)


class AudioSample:
    def __init__(self, data: AudioSegment):
        self.data = data

    def __len__(self):
        return self.n_frames

    @property
    def n_frames(self) -> int:
        return int(self.data.frame_count())

    @property
    def n_seconds(self) -> float:
        return self.data.duration_seconds

    def slice_time(self, start_stop: TimeType, stop: Optional[TimeType] = None) -> 'AudioSample':
        """
        Both ``start_stop`` and ``stop`` are in seconds
        """
        if stop is None:
            start = 0
            stop = start_stop
        else:
            start = start_stop

        # pydub slices in milliseconds
        start_ms = start * 1000
        stop_ms = stop * 1000
        return AudioSample(self.data[start_ms:stop_ms])

    def slice_frame(self, start_stop: int, stop: Optional[int] = None) -> 'AudioSample':
        if stop is None:
            start = 0
            stop = start_stop
        else:
            start = start_stop

        # pydub slices in milliseconds
        start = self.frame_to_seconds(start)
        stop = self.frame_to_seconds(stop)
        return self.slice_time(start, stop)

    def to_bytes(self) -> bytes:
        return self.data.raw_data

    def to_numpy(self) -> np.ndarray:
        return np.frombuffer(self.to_bytes(), dtype=f'int{self.bit_depth}')

    def to_array(self) -> array:
        return self.data.get_array_of_samples()

    def seconds_to_frame(self, seconds: TimeType) -> int:
        return int(seconds * self.frame_rate)

    def frame_to_seconds(self, frame: int) -> float:
        return frame / self.frame_rate

    @property
    def frame_rate(self) -> int:
        return self.data.frame_rate

    @property
    def n_channels(self) -> int:
        return self.data.channels

    @property
    def bit_depth(self) -> int:
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
        return (2 ** self.bit_depth) / 2  # The "/2" is because half of the size is for above 0, the other for below

    def convert(self, frame_rate: Optional[int] = None, sample_width: Optional[int] = None, n_channels: Optional[int] = None) -> 'AudioSample':
        new_data = self.data

        if frame_rate is not None:
            new_data = new_data.set_frame_rate(frame_rate)

        if sample_width is not None:
            new_data = new_data.set_sample_width(sample_width)

        if n_channels is not None:
            new_data = new_data.set_channels(n_channels)

        return AudioSample(new_data)

    def invert_phase(self) -> 'AudioSample':
        return AudioSample(self.data.invert_phase())

    def normalize(self) -> 'AudioSample':
        return AudioSample(self.data.normalize(0))

    def append(self, audio_sample: 'AudioSample', crossfade: TimeType = 0) -> 'AudioSample':
        return AudioSample(self.data.append(audio_sample.data, crossfade=crossfade))

    def overlay(self, audio_sample: 'AudioSample', offset: TimeType = 0) -> 'AudioSample':
        return AudioSample(
            self.data.overlay(audio_sample.data, position=int(offset * 1000))  # pydub works in ms
        )

    @classmethod
    def from_iterable(cls, audio_samples: Iterable['AudioSample'], crossfade: TimeType = 0) -> 'AudioSample':
        final_sample = None
        for sample in audio_samples:
            if final_sample is None:
                final_sample = sample
            else:
                final_sample = final_sample.append(sample, crossfade=crossfade)
        return final_sample

    @classmethod
    def from_numpy_and_sample(cls, data: np.ndarray, source_sample: 'AudioSample') -> 'AudioSample':
        raise NotImplementedError('The implementation loses half of the audio')
        data_array = array(source_sample.data.array_type, data)
        return cls.from_array_and_sample(data_array, source_sample)

    @classmethod
    def from_array_and_sample(cls, data: array, source_sample: 'AudioSample') -> 'AudioSample':
        raise NotImplementedError('The implementation loses half of the audio')
        return source_sample.data._spawn(data)

    def play(self, delta_gain_dB=None) -> None:
        data = self.data
        if delta_gain_dB is not None:
            data = data.apply_gain(delta_gain_dB)
        play(data)

    def export(self, filename, **kwargs) -> io.BufferedRandom:
        return self.data.export(filename, **kwargs)

    def remove_dc_offset(self, rolling: bool = False) -> 'AudioSample':
        if not rolling:
            return AudioSample(
                self.data.remove_dc_offset()
            )
        '''
        # This should work, but I lose half the audio when converting the numpy array back into an AudioSegment
        window_size = 15

        samples = self.to_numpy()
        view = sliding_window_view(samples, window_size)
        frame_averages = np.concatenate((
            view.mean(axis=-1).astype(int),
            [int(samples[-(window_size-1):].mean())]*(window_size-1)
        ))

        adjusted = samples - frame_averages
        return AudioSample.from_numpy_and_sample(adjusted, self)
        '''

        samples = self.to_numpy()

        long_duration_threshold = 2000

        is_negative = samples < 0
        sign_change = np.concatenate((
            [is_negative[0]],
            is_negative[:-1] != is_negative[1:],
            [True]
        ))
        indices_where_sign_changed = np.where(sign_change)[0]
        duration_of_groups = np.diff(indices_where_sign_changed)

        is_long_duration = duration_of_groups > long_duration_threshold
        long_durations = np.concatenate((is_long_duration, [False]))
        start_of_long_durations = indices_where_sign_changed[long_durations]
        end_of_long_durations = indices_where_sign_changed[1:][long_durations[:-1]]

        final_audio_segment = self
        for start, end in zip(start_of_long_durations, end_of_long_durations):
            inverted_region = self.slice_frame(start, end).invert_phase()
            final_audio_segment = final_audio_segment.overlay(
                inverted_region,
                offset=start
            )

        # TODO: There may still be short spikes to eliminate, but I'm not sure how to do that without going into numpy
        #  and converting back to AudioSegment doesn't seem to work properly

        return final_audio_segment

    def plot_amplitude(
            self,
            *,
            title=None,
            axis=None,
            highlight_regions: Iterable[Tuple[TimeType, TimeType]] = (),
            normalize_amplitude: bool = False,
    ) -> None:
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
        axis.set_xlabel('Time (sec)')
        axis.set_ylabel('Amplitude')

        amplitude_line, *_ = axis.plot(time, samples)
        amplitude_color = amplitude_line.get_color()

        for region in highlight_regions:
            axis.fill_between(region, samples.min(), samples.max(), facecolor=amplitude_color, alpha=0.3)

        if show_plot:
            plt.show()

    plot = plot_amplitude

    def plot_spectrogram(self, title=None, *, mode='psd', axis=None):
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
        fig, (ax_amplitude, ax_spectrogram) = plt.subplots(nrows=2)

        self.plot_amplitude(axis=ax_amplitude)
        self.plot_spectrogram(axis=ax_spectrogram)

        plt.show()
