import io
from typing import Union, Optional

from pydub import AudioSegment
from pydub.playback import play


class BaseAudioSource:
    @property
    def frame_rate(self) -> int:
        raise NotImplementedError

    def seconds_to_frames(self, seconds: Union[float, int]) -> int:
        return int(seconds * self.frame_rate)

    def read(self, n_frames: Optional[int]) -> 'AudioSample':
        raise NotImplementedError

    def read_seconds(self, n_seconds: Union[int, float, None]) -> 'AudioSample':
        assert n_seconds is None or (isinstance(n_seconds, (int, float)) and n_seconds > 0), f'n_seconds must be None or a number greater than zero, got {n_seconds!r}'

        if n_seconds is None:
            n_frames = None
        else:
            n_frames = self.seconds_to_frames(n_seconds)
        return self.read(n_frames)


class AudioSample:
    def __init__(self, data: AudioSegment):
        self.data = data

    @property
    def raw_data(self) -> bytes:
        return self.data.raw_data

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

    def convert(self, frame_rate: Optional[int] = None, frame_width: Optional[int] = None, n_channels: Optional[int] = None) -> 'AudioSample':
        new_data = self.data

        if frame_rate is not None:
            new_data = new_data.set_frame_rate(frame_rate)

        if frame_width is not None:
            new_data = new_data.set_sample_width(frame_width)

        if n_channels is not None:
            new_data = new_data.set_channels(n_channels)

        return AudioSample(new_data)

    def play(self, delta_gain_dB=None) -> None:
        data = self.data
        if delta_gain_dB is not None:
            data = data.apply_gain(delta_gain_dB)
        play(data)

    def export(self, filename, **kwargs) -> io.BufferedRandom:
        return self.data.export(filename, **kwargs)
