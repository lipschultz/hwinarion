import json
from pathlib import Path
from typing import Union

from vosk import Model, KaldiRecognizer

from hwinarion.audio.base import AudioSample
from hwinarion.speech_to_text import BaseSpeechToText
from hwinarion.speech_to_text.base import DetailedTranscript, TranscriptSegment, DetailedTranscripts


class VoskSpeechToText(BaseSpeechToText):
    FRAME_RATE = 16_000
    BIT_DEPTH = 16
    N_CHANNELS = 1

    def __init__(self, model_path: Union[str, Path], frame_rate: int = FRAME_RATE, bit_depth: int = BIT_DEPTH,
                 n_channels: int = N_CHANNELS):
        super().__init__()
        self.model_path = model_path
        self.frame_rate = frame_rate
        self.bit_depth = bit_depth
        self.n_channels = n_channels

        self._model = Model(self.model_path)
        self._recognizer = KaldiRecognizer(self._model, self.frame_rate)

    @property
    def sample_width(self) -> int:
        return self.bit_depth // 8

    def transcribe_audio_detailed(self, audio_data: AudioSample, *, n_transcriptions: int = 3, segment_timestamps: bool = True) -> DetailedTranscripts:
        """
        Transcribe an audio sample, returning a transcript with extra details about the transcription, such as
        timestamps and confidence.

        ``n_transcriptions`` indicates the maximum number of transcripts to generate.

        ``segment_timestamps``, if True, will provide start and end timestamps for each word in the transcript.
        """
        self._recognizer.SetMaxAlternatives(n_transcriptions)
        self._recognizer.SetWords(segment_timestamps)
        self._recognizer.AcceptWaveform(
            audio_data.convert(
                sample_width=self.sample_width,
                frame_rate=self.frame_rate,
                n_channels=self.n_channels
            ).to_bytes()
        )
        result = json.loads(self._recognizer.FinalResult())

        return DetailedTranscripts(
            [
                DetailedTranscript(
                    transcript['text'],
                    transcript['confidence'],
                    [
                        TranscriptSegment(segment['word'], segment['start'], segment['end'])
                        for segment in transcript['result']
                    ] if 'result' in transcript else None
                )
                for transcript in result['alternatives']
            ]
        )
