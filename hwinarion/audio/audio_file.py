from pathlib import Path
from typing import Union, Optional

from pydub import AudioSegment

from hwinarion.audio.base import BaseAudioSource, AudioSample


class AudioFile(BaseAudioSource):
    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)
        self._audio_data = AudioSegment.from_file(filepath)
        self.frame_index = 0

    def __max_audio_frames(self) -> int:
        return int(self._audio_data.frame_count())

    def jump_to_frame(self, frame_number: int):
        assert isinstance(frame_number, int) and 0 <= frame_number < self.__max_audio_frames()
        self.frame_index = frame_number

    def reset(self):
        self.jump_to_frame(0)

    @property
    def frame_rate(self) -> int:
        return self._audio_data.frame_rate

    def read_pydub(self, n_frames: Optional[int]) -> AudioSegment:
        if self.frame_index >= self.__max_audio_frames():
            raise EOFError(f'Attempted to read past the end of the audio file.')

        if n_frames is None:
            n_frames = self.__max_audio_frames() - self.frame_index

        assert isinstance(n_frames, int) and n_frames > 0, f'n_frames must be an integer greater than zero, got {n_frames!r}'

        read_data = self._audio_data.get_sample_slice(self.frame_index, self.frame_index + n_frames)
        self.frame_index += n_frames
        return read_data

    def read(self, n_frames: Optional[int]) -> AudioSample:
        """
        Read frames of audio from the file and return them in an ``AudioSample``.

        If ``n_frames` is ``None``, read all remaining frames in the file.
        """
        return AudioSample(self.read_pydub(n_frames))



