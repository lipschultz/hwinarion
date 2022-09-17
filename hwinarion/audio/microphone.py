import io
from typing import Optional, List

import pyaudio
from pydub import AudioSegment

from hwinarion.audio.base import BaseAudioSource, AudioSample


class Microphone(BaseAudioSource):
    def __init__(self, device_index: Optional[int] = None, chunk_size: int = 1024):
        """
        The ``device_index`` is used to tell PyAudio which audio device to listen on.

        ``chunk_size`` is the number of frames to store in a buffer.  A larger buffer reduces the chance of triggering when ambient noise changes rapidly.
        """
        pa = pyaudio.PyAudio()

        assert device_index is None or (isinstance(device_index, int) and 0 <= device_index < pa.get_device_count()), f'device_index must be None or positive integer between 0 and {pa.get_device_count()}, got: {device_index!r}'
        assert isinstance(chunk_size, int) and chunk_size > 0, f'chunk_size must be a positive integer, got: {chunk_size!r}'

        self._device_index = device_index
        self.chunk_size = chunk_size
        self.DEFAULT_READ_DURATION_SECONDS = 5

    @classmethod
    def get_device_names(cls) -> List[str]:
        pa = pyaudio.PyAudio()
        try:
            return [
                pa.get_device_info_by_index(i).get('name')
                for i in range(pa.get_device_count())
            ]
        finally:
            pa.terminate()

    @property
    def device_index(self) -> Optional[int]:
        return self._device_index

    @property
    def audio_device_information(self) -> dict:
        pa = pyaudio.PyAudio()
        try:
            if self.device_index is None:
                return pa.get_default_input_device_info()
            else:
                return pa.get_device_info_by_index(self.device_index)
        finally:
            pa.terminate()

    @property
    def frame_rate(self) -> int:
        return int(self.audio_device_information['defaultSampleRate'])

    @property
    def n_channels(self) -> int:
        return 1

    @property
    def _pyaudio_format(self) -> int:
        return pyaudio.paInt16

    @property
    def bit_depth(self) -> int:
        return 16

    @property
    def frame_width(self) -> int:
        return pyaudio.get_sample_size(self._pyaudio_format)

    @property
    def DEFAULT_READ_DURATION_FRAMES(self) -> int:
        return self.seconds_to_frames(self.DEFAULT_READ_DURATION_SECONDS)

    def read_bytes(self, n_frames: int) -> bytes:
        assert isinstance(n_frames, int) and n_frames > 0, f'n_frames must be an integer greater than zero, got {n_frames!r}'

        pa = pyaudio.PyAudio()
        try:
            audio_source = pa.open(
                input_device_index=self.device_index,
                channels=self.n_channels,
                format=self._pyaudio_format,
                rate=self.frame_rate,
                frames_per_buffer=self.chunk_size,
                input=True,
            )
            return audio_source.read(n_frames, exception_on_overflow=False)
        finally:
            pa.terminate()

    def read_pydub(self, n_frames: int) -> AudioSegment:
        with io.BytesIO(self.read_bytes(n_frames)) as fp:
            return AudioSegment.from_raw(
                fp,
                sample_width=self.frame_width,
                frame_rate=self.frame_rate,
                channels=self.n_channels
            )

    def read(self, n_frames: Optional[int]) -> AudioSample:
        if n_frames is None:
            n_frames = self.DEFAULT_READ_DURATION_FRAMES
        return AudioSample(self.read_pydub(n_frames))
