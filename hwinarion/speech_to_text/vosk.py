import json
from pathlib import Path
from typing import Union

from vosk import Model, KaldiRecognizer, SetLogLevel

from hwinarion.speech_to_text import BaseSpeechToText


class VoskSpeechToText(BaseSpeechToText):
    FRAME_RATE = 16_000
    BIT_DEPTH = 16
    N_CHANNELS = 1

    def __init__(self, model_path: Union[str, Path], frame_rate: int = FRAME_RATE, bit_depth: int = BIT_DEPTH, n_channels: int = N_CHANNELS):
        super().__init__()
        self.model_path = model_path
        self.frame_rate = frame_rate
        self.bit_depth = bit_depth
        self.n_channels = n_channels

        self._model = Model(self.model_path)
        self._recognizer = KaldiRecognizer(self._model, self.frame_rate)

    @property
    def frame_width(self) -> int:
        return self.bit_depth // 8

    def transcribe_audio(self, audio_data) -> str:
        self._recognizer.AcceptWaveform(
            audio_data.convert(frame_width=self.frame_width, frame_rate=self.frame_rate, n_channels=self.n_channels).raw_data
        )
        return json.loads(self._recognizer.FinalResult())['text']
