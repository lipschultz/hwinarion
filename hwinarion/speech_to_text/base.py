from dataclasses import dataclass
from typing import List, Optional

from hwinarion.audio.base import AudioSample


class NoTranscriptionError(Exception):
    def __init__(self, msg=None, audio_file=None):
        super().__init__(msg, audio_file)


@dataclass(frozen=True)
class TranscriptSegment:
    text: str
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass(frozen=True)
class DetailedTranscript:
    text: str
    confidence: Optional[float]
    segments: Optional[List[TranscriptSegment]]


@dataclass(frozen=True)
class DetailedTranscripts:
    transcripts: List[DetailedTranscript]

    def __post_init__(self):
        self.transcripts.sort(key=lambda detailed_transcript: detailed_transcript.confidence, reverse=True)

    @property
    def best_transcript(self) -> DetailedTranscript:
        return self.transcripts[0]


class BaseSpeechToText:
    def __init__(self):
        pass

    def transcribe_audio_detailed(
        self, audio: AudioSample, *, n_transcriptions: int = 3, segment_timestamps: bool = True
    ) -> DetailedTranscripts:
        raise NotImplementedError

    def transcribe_audio(self, audio: AudioSample) -> str:
        return self.transcribe_audio_detailed(audio, n_transcriptions=1).best_transcript.text
