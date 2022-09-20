import io
from dataclasses import dataclass
from enum import Enum
from typing import Union, Optional, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from pydub.playback import play


TimeType = Union[float, int]


class BaseAudioSource:
    @property
    def frame_rate(self) -> int:
        raise NotImplementedError

    def seconds_to_frames(self, seconds: TimeType) -> int:
        return int(seconds * self.frame_rate)

    def read(self, n_frames: Optional[int]) -> 'AudioSample':
        raise NotImplementedError

    def read_seconds(self, n_seconds: Optional[TimeType]) -> 'AudioSample':
        assert n_seconds is None or (isinstance(n_seconds, (int, float)) and n_seconds > 0), f'n_seconds must be None or a number greater than zero, got {n_seconds!r}'

        if n_seconds is None:
            n_frames = None
        else:
            n_frames = self.seconds_to_frames(n_seconds)
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

    @property
    def raw_data(self) -> bytes:
        return self.data.raw_data

    def to_numpy(self) -> np.ndarray:
        return np.frombuffer(self.raw_data, dtype=f'int{self.bit_depth}')

    @property
    def frame_rate(self) -> int:
        return self.data.frame_rate

    @property
    def n_channels(self) -> int:
        return self.data.channels

    @property
    def bit_depth(self) -> int:
        return self.frame_width * 8

    @property
    def frame_width(self) -> int:
        return self.data.frame_width

    @property
    def rms(self) -> int:
        """
        A measure of the loudness or energy in the audio.
        """
        return self.data.rms

    def convert(self, frame_rate: Optional[int] = None, frame_width: Optional[int] = None, n_channels: Optional[int] = None) -> 'AudioSample':
        new_data = self.data

        if frame_rate is not None:
            new_data = new_data.set_frame_rate(frame_rate)

        if frame_width is not None:
            new_data = new_data.set_sample_width(frame_width)

        if n_channels is not None:
            new_data = new_data.set_channels(n_channels)

        return AudioSample(new_data)

    def append(self, audio_sample: 'AudioSample', crossfade: TimeType = 0) -> 'AudioSample':
        return AudioSample(self.data.append(audio_sample.data, crossfade=crossfade))

    @classmethod
    def from_iterable(cls, audio_samples: Iterable['AudioSample'], crossfade: TimeType = 0) -> 'AudioSample':
        final_sample = None
        for sample in audio_samples:
            if final_sample is None:
                final_sample = sample
            else:
                final_sample = final_sample.append(sample, crossfade=crossfade)
        return final_sample

    def play(self, delta_gain_dB=None) -> None:
        data = self.data
        if delta_gain_dB is not None:
            data = data.apply_gain(delta_gain_dB)
        play(data)

    def export(self, filename, **kwargs) -> io.BufferedRandom:
        return self.data.export(filename, **kwargs)

    def plot_amplitude(self, title=None, *, axis=None):
        samples = self.to_numpy()
        time = np.linspace(0, self.n_seconds, num=len(self))

        show_plot = axis is None
        if axis is None:
            _, axis = plt.subplots()

        if title is not None:
            axis.set_title(title)
        axis.set_xlabel('Time (sec)')
        axis.set_ylabel('Amplitude')
        axis.plot(time, samples)

        if show_plot:
            plt.show()

    def plot_spectrogram(self, title=None, *, mode='psd', axis=None):
        samples = self.to_numpy()

        show_plot = axis is None
        if axis is None:
            _, axis = plt.subplots()

        if title is not None:
            axis.set_title(title)
        axis.specgram(samples)

        if show_plot:
            plt.show()

    def plot(self):
        samples = self.to_numpy()
        time = np.linspace(0, self.n_seconds, num=len(self))

        fig, (ax_amplitude, ax_spectrogram) = plt.subplots(nrows=2)

        self.plot_amplitude(axis=ax_amplitude)
        self.plot_spectrogram(axis=ax_spectrogram)

        plt.show()


FrameStateEnum = Enum('FrameStateEnum', ('LISTEN', 'PAUSE', 'STOP'))


@dataclass
class AnnotatedFrame:
    frame: AudioSample
    state: FrameStateEnum


class BaseListener:
    def __init__(self, source: BaseAudioSource):
        self.source = source

    def _determine_frame_state(self, latest_frame: AudioSample, all_frames: List[AnnotatedFrame]) -> FrameStateEnum:
        raise NotImplementedError

    def _pre_process_individual_audio_sample(self, audio: AudioSample) -> AudioSample:
        return audio

    def _filter_audio_samples(self, all_frames: List[AnnotatedFrame]) -> List[AnnotatedFrame]:
        previous_state = None
        saved_frames = []
        for frame in all_frames:
            if frame.state == FrameStateEnum.LISTEN or previous_state == FrameStateEnum.LISTEN:
                saved_frames.append(frame)
            previous_state = frame.state
        return saved_frames

    def _join_audio_samples(self, all_frames: List[AnnotatedFrame]) -> AudioSample:
        return AudioSample.from_iterable(frame.frame for frame in all_frames)

    def _post_process_final_audio_sample(self, audio: AudioSample) -> AudioSample:
        return audio

    def listen_frames(self, chunk_size: int = 2**14) -> List[AnnotatedFrame]:
        all_frames = []

        frame = self.source.read(chunk_size)
        frame = self._pre_process_individual_audio_sample(frame)
        frame_state = self._determine_frame_state(frame, all_frames)

        while frame_state != FrameStateEnum.STOP:
            all_frames.append(AnnotatedFrame(frame, frame_state))
            frame = self.source.read(chunk_size)
            frame = self._pre_process_individual_audio_sample(frame)
            frame_state = self._determine_frame_state(frame, all_frames)

        all_frames.append(AnnotatedFrame(frame, frame_state))
        return all_frames

    def listen(self, chunk_size: int = 2**14) -> AudioSample:
        """
        If the end of the source is reached, then listening will end regardless of what ``_determine_frame_state`` will return.
        """
        all_frames = self.listen_frames(chunk_size)
        all_frames = self._filter_audio_samples(all_frames)
        audio_sample = self._join_audio_samples(all_frames)
        audio_sample = self._post_process_final_audio_sample(audio_sample)

        return audio_sample


class TimeBasedListener(BaseListener):
    def __init__(self, source: BaseAudioSource, total_duration: TimeType):
        super().__init__(source)
        self.total_duration = total_duration

    def _determine_frame_state(self, latest_frame: AudioSample, all_frames: List[AnnotatedFrame]) -> FrameStateEnum:
        duration = sum(frame.frame.n_seconds for frame in all_frames) + latest_frame.n_seconds
        if duration < self.total_duration and len(latest_frame) > 0:
            return FrameStateEnum.LISTEN
        else:
            return FrameStateEnum.STOP


class SilenceBasedListener(BaseListener):
    def __init__(self, source: BaseAudioSource, silence_threshold_rms: int = 500):
        super().__init__(source)
        self.silence_threshold_rms = silence_threshold_rms

    def _determine_frame_state(self, latest_frame: AudioSample, all_frames: List[AnnotatedFrame]) -> FrameStateEnum:
        if latest_frame.rms > self.silence_threshold_rms:
            print(len(all_frames), latest_frame.rms, 'LISTEN')
            return FrameStateEnum.LISTEN
        else:
            if len(all_frames) == 0 or all_frames[-1].state == FrameStateEnum.PAUSE:
                # Haven't started listening yet
                print(len(all_frames), latest_frame.rms,'PAUSE')
                return FrameStateEnum.PAUSE
            else:
                print(len(all_frames), latest_frame.rms,'STOP')
                return FrameStateEnum.STOP


class Listener:
    def __init__(self, source: BaseAudioSource, speaking_threshold_rms: int = 300):
        self.source = source
        self.speaking_threshold_rms = speaking_threshold_rms

    def contains_speaking(self, audio_sample: AudioSample) -> bool:
        return audio_sample.rms >= self.speaking_threshold_rms

    def listen(self, start_timeout: Optional[TimeType] = None, phrase_time_limit=None) -> 'AudioSample':
        """
        Records a single phrase from ``source`` (an ``AudioSource`` instance) into an ``AudioData`` instance, which it returns.

        This is done by waiting until the audio has an energy above ``recognizer_instance.energy_threshold`` (the user has started speaking), and then recording until it encounters ``recognizer_instance.pause_threshold`` seconds of non-speaking or there is no more audio input. The ending silence is not included.

        The ``start_timeout`` parameter is the maximum number of seconds that this will wait for a phrase to start before giving up and throwing an ``speech_recognition.WaitTimeoutError`` exception. If ``start_timeout`` is ``None``, there will be no wait timeout.

        The ``phrase_time_limit`` parameter is the maximum number of seconds that this will allow a phrase to continue before stopping and returning the part of the phrase processed before the time limit was reached. The resulting audio will be the phrase cut off at the time limit. If ``phrase_timeout`` is ``None``, there will be no phrase time limit.

        This operation will always complete within ``start_timeout + phrase_timeout`` seconds if both are numbers, either by returning the audio data, or by raising a ``speech_recognition.WaitTimeoutError`` exception.
        """
        assert self.pause_threshold >= self.non_speaking_duration >= 0

        chunk_size = 1024

        seconds_per_buffer = chunk_size / self.source.frame_rate
        pause_buffer_count = int(math.ceil(self.pause_threshold / seconds_per_buffer))  # number of buffers of non-speaking audio during a phrase, before the phrase should be considered complete
        phrase_buffer_count = int(math.ceil(self.phrase_threshold / seconds_per_buffer))  # minimum number of buffers of speaking audio before we consider the speaking audio a phrase
        non_speaking_buffer_count = int(math.ceil(self.non_speaking_duration / seconds_per_buffer))  # maximum number of buffers of non-speaking audio to retain before and after a phrase

        # read audio input for phrases until there is a phrase that is long enough
        elapsed_time = 0
        while True:
            frames = collections.deque()

            # Loop until speaking starts (or start_timeout is reached or end of source)
            while True:
                frame_buffer = self.source.read(chunk_size)
                elapsed_time += frame_buffer.n_seconds
                if len(frame_buffer) == 0:
                    break  # end of source reached

                frames.append(frame_buffer)
                if len(frames) > non_speaking_buffer_count:  # ensure we only keep the needed amount of non-speaking buffers
                    frames.popleft()

                if self.contains_speaking(frame_buffer):
                    break

                if start_timeout and elapsed_time > start_timeout:
                    raise WaitTimeoutError("listening timed out while waiting for phrase to start")

                """
                # dynamically adjust the energy threshold using asymmetric weighted average
                if self.dynamic_energy_threshold:
                    damping = self.dynamic_energy_adjustment_damping ** seconds_per_buffer  # account for different chunk sizes and rates
                    target_energy = energy * self.dynamic_energy_ratio
                    self.energy_threshold = self.energy_threshold * damping + target_energy * (1 - damping)
                """

            # Loop until speaking ends (or phrase_time_limit is reached)
            phrase_count = 0
            pause_count = 0
            phrase_start_time = elapsed_time
            while True:
                frame_buffer = source.stream.read(chunk_size)
                elapsed_time += frame_buffer.n_seconds
                if len(frame_buffer) == 0:
                    break  # end of source reached

                frames.append(frame_buffer)
                phrase_count += 1

                # check if speaking has stopped for longer than the pause threshold on the audio input
                if self.contains_speaking(frame_buffer):
                    pause_count = 0
                else:
                    pause_count += 1

                if pause_count > pause_buffer_count:  # end of the phrase
                    break

                # handle phrase being too long by cutting off the audio
                if phrase_time_limit and elapsed_time - phrase_start_time > phrase_time_limit:
                    break

            # check how long the detected phrase is, and retry listening if the phrase is too short
            phrase_count -= pause_count  # exclude the buffers for the pause before the phrase
            if phrase_count >= phrase_buffer_count or len(buffer) == 0: break  # phrase is long enough or we've reached the end of the stream, so stop listening

        # obtain frame data
        for i in range(pause_count - non_speaking_buffer_count): frames.pop()  # remove extra non-speaking frames at the end
        frame_data = b"".join(frames)

        return AudioData(frame_data, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
