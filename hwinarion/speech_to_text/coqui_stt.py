from pathlib import Path
from typing import Union, Optional, Tuple, Dict

from stt import Model
from stt.impl import CandidateTranscript

from hwinarion.audio.base import AudioSample
from hwinarion.speech_to_text import BaseSpeechToText
from hwinarion.speech_to_text.base import DetailedTranscript, TranscriptSegment, DetailedTranscripts


class CoquiSpeechToText(BaseSpeechToText):
    def __init__(
            self,
            model_path: Union[str, Path],
            beam_width: Optional[int] = None,
            external_scorer_path: Union[str, Path, None] = None,
            lm_alpha_beta: Optional[Tuple[float, float]] = None,
            hotword_boost: Optional[Dict[str, float]] = None,
            n_channels: int = 1
    ):
        super().__init__()
        self.model_path = model_path
        self.beam_width = beam_width
        self.external_scorer_path = external_scorer_path
        self.hotword_boost = hotword_boost or {}
        self.n_channels = n_channels

        self._model = Model(str(self.model_path))

        if self.beam_width is not None:
            self._model.setBeamWidth(self.beam_width)

        if self.external_scorer_path is not None:
            self._model.enableExternalScorer(str(self.external_scorer_path))
            if lm_alpha_beta is not None:
                self._model.setScorerAlphaBeta(lm_alpha_beta[0], lm_alpha_beta[1])

        if self.hotword_boost:
            for word, boost in hotword_boost.items():
                self._model.addHotWord(word, boost)

    @property
    def frame_rate(self) -> int:
        return self._model.sampleRate()

    def transcribe_audio_detailed(self, audio_data: AudioSample, *, n_transcriptions: int = 3, segment_timestamps: bool = True) -> DetailedTranscripts:
        result = self._model.sttWithMetadata(
            audio_data.convert(
                frame_rate=self.frame_rate,
                n_channels=self.n_channels,
            ).to_numpy(),
            n_transcriptions
        )

        results = DetailedTranscripts(
            [
                self._coqui_token_metadata_to_detailed_transcript(transcript, segment_timestamps=segment_timestamps)
                for transcript in result.transcripts
            ]
        )
        return results

    @staticmethod
    def _coqui_token_metadata_to_detailed_transcript(transcript: CandidateTranscript, segment_timestamps: bool) -> DetailedTranscript:
        transcript_segments = []
        word = []
        start_time = None
        for i, token in enumerate(transcript.tokens):
            if len(word) == 0:
                word.append(token.text)
                start_time = token.start_time
            elif token == ' ' or i == len(transcript.tokens) - 1:
                transcript_segments.append(TranscriptSegment(
                    ''.join(word),
                    start_time,
                    token.start_time,
                ))
                word = []
                start_time = None
            else:
                word.append(token.text)

        return DetailedTranscript(
            ' '.join(segment.text for segment in transcript_segments),
            transcript.confidence,
            transcript_segments if segment_timestamps else None
        )
