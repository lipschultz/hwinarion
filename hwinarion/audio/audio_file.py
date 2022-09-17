from pathlib import Path
from typing import Union, Optional

from pydub import AudioSegment

from hwinarion.audio.base import BaseAudioSource, AudioSample


class AudioFile(BaseAudioSource):
    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)
        self._audio_data = AudioSegment.from_file(filepath)
        self.frame_index = 0

    @property
    def frame_rate(self) -> int:
        return self._audio_data.frame_rate

    def read_pydub(self, n_frames: Optional[int]) -> AudioSegment:
        if n_frames is None:
            return self._audio_data

        assert isinstance(n_frames, int) and n_frames > 0, f'n_frames must be an integer greater than zero, got {n_frames!r}'

        read_data = self._audio_data.get_sample_slice(self.frame_index, self.frame_index + n_frames)
        self.frame_index += n_frames
        return read_data

    def read(self, n_frames: Optional[int]) -> AudioSample:
        return AudioSample(self.read_pydub(n_frames))



