import os
from pathlib import Path
from typing import Union

from pocketsphinx import pocketsphinx

from hwinarion.audio.base import AudioSample
from hwinarion.speech_to_text.base import (
    BaseSpeechToText,
    DetailedTranscript,
    DetailedTranscripts,
    NoTranscriptionError,
    TranscriptSegment,
)


class SphinxSpeechToText(BaseSpeechToText):
    FRAME_RATE = 16_000
    BIT_DEPTH = 16
    N_CHANNELS = 1

    START_TOKEN = "<s>"
    END_TOKEN = "</s>"
    SILENCE_TOKEN = "<sil>"

    def __init__(
        self,
        model_path: Union[str, Path],
        frame_rate: int = FRAME_RATE,
        bit_depth: int = BIT_DEPTH,
        n_channels: int = N_CHANNELS,
    ):
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

    def transcribe_audio_detailed(
        self,
        audio: AudioSample,
        *,
        n_transcriptions: int = 1,
        segment_timestamps: bool = True,
    ) -> DetailedTranscripts:
        assert n_transcriptions == 1

        self._recognizer.start_utt()
        self._recognizer.process_raw(
            audio.convert(
                sample_width=self.sample_width,
                frame_rate=self.frame_rate,
                n_channels=self.n_channels,
            ).to_bytes(),
            False,  # no search
            True,  # full utterance
        )
        self._recognizer.end_utt()

        hypothesis = self._recognizer.hyp()
        if hypothesis is None:
            raise NoTranscriptionError(audio_file=audio)

        # Sphinx docs imply fps is 100, but experimentation showed that wasn't the case
        sphinx_fps = list(self._recognizer.seg())[-1].end_frame / audio.n_seconds if segment_timestamps else None

        return DetailedTranscripts(
            [
                DetailedTranscript(
                    hypothesis.hypstr,
                    hypothesis.prob,
                    [
                        TranscriptSegment(
                            segment.word,
                            segment.start_frame / sphinx_fps,
                            segment.end_frame / sphinx_fps,
                        )
                        for segment in self._recognizer.seg()
                    ]
                    if segment_timestamps
                    else None,
                )
            ],
            {"hypothesis": hypothesis},
        )
