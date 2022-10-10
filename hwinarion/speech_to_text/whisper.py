import tempfile
from pathlib import Path
from typing import Union, Optional

import torch
import whisper

from hwinarion.audio.base import AudioSample
from hwinarion.speech_to_text import BaseSpeechToText
from hwinarion.speech_to_text.base import DetailedTranscript, DetailedTranscripts, TranscriptSegment


class WhisperSpeechToText(BaseSpeechToText):
    def __init__(
        self,
        model_name: str,
        device: Optional[Union[str, torch.device]] = None,
        download_root: Union[str, Path] = None,
        in_memory: bool = False
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.download_root = download_root
        self.in_memory = in_memory

        self._model = whisper.load_model(self.model_name, self.device, str(self.download_root), self.in_memory)

    def transcribe_audio_detailed(
        self,
        audio_data: AudioSample,
        *,
        n_transcriptions: int = 3,
        segment_timestamps: bool = True,
    ) -> DetailedTranscripts:
        """
        Transcribe an audio sample, returning a transcript with extra details about the transcription, such as
        timestamps and confidence.

        ``n_transcriptions`` indicates the maximum number of transcripts to generate.

        ``segment_timestamps``, if True, will provide start and end timestamps for each word in the transcript.
        """
        with tempfile.NamedTemporaryFile(suffix='.wav') as temp_fp:
            audio_data.export(temp_fp.name)
            result = self._model.transcribe(temp_fp.name)

        return DetailedTranscripts(
            [
                DetailedTranscript(
                    result["text"],
                    0,
                    [
                        TranscriptSegment(segment['text'], segment['start'], segment['end'])
                        for segment in result['segments']
                    ]
                )
            ]
        )
