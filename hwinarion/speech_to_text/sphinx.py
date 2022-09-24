import os
from pathlib import Path
from typing import Union

from pocketsphinx import pocketsphinx

from hwinarion.audio.base import AudioSample
from hwinarion.speech_to_text import BaseSpeechToText
from hwinarion.speech_to_text.base import NoTranscriptionError, DetailedTranscripts, DetailedTranscript


class SphinxSpeechToText(BaseSpeechToText):
    FRAME_RATE = 16_000
    BIT_DEPTH = 16
    N_CHANNELS = 1

    def __init__(self, model_path: Union[str, Path], frame_rate: int = FRAME_RATE, bit_depth: int = BIT_DEPTH, n_channels: int = N_CHANNELS):
        super().__init__()
        self.model_path = Path(model_path)
        self.frame_rate = frame_rate
        self.bit_depth = bit_depth
        self.n_channels = n_channels

        acoustic_model_path = self.model_path / "acoustic-model"
        language_model_path = self.model_path / "language-model.lm.bin"
        phoneme_dictionary_path = self.model_path / "pronounciation-dictionary.dict"

        config = pocketsphinx.Decoder.default_config()
        config.set_string("-hmm", str(acoustic_model_path))
        config.set_string("-lm", str(language_model_path))
        config.set_string("-dict", str(phoneme_dictionary_path))
        config.set_string("-logfn", os.devnull)  # Disables logging
        self._recognizer = pocketsphinx.Decoder(config)

    @property
    def sample_width(self) -> int:
        return self.bit_depth // 8

    def transcribe_audio_detailed(self, audio_data: AudioSample, search=False, full_utt=True) -> DetailedTranscripts:
        self._recognizer.start_utt()
        self._recognizer.process_raw(
            audio_data.convert(
                sample_width=self.sample_width,
                frame_rate=self.frame_rate,
                n_channels=self.n_channels
            ).to_bytes(),
            search,  # no search
            full_utt  # full utterance
        )
        self._recognizer.end_utt()

        hypothesis = self._recognizer.hyp()
        if hypothesis is None:
            raise NoTranscriptionError(audio_file=audio_data)

        return DetailedTranscripts(
            [
                DetailedTranscript(
                    hypothesis.hypstr,
                    hypothesis.prob,
                    None
                )
            ]
        )
