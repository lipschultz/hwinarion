from unittest import mock

import pytest

from hwinarion.audio.base import AudioSample
from hwinarion.speech_to_text import base


@pytest.mark.parametrize("start_time, end_time, expected_duration", [
    (0, 1, 1),
    (3, 10, 7),
    (3, 3, 0),
])
def test_transcript_segment_duration(start_time, end_time, expected_duration):
    subject = base.TranscriptSegment('any text', start_time, end_time)

    actual_duration = subject.duration

    assert expected_duration == actual_duration


@pytest.mark.parametrize("transcripts, expected_best_transcript", [
    (
        [
            base.DetailedTranscript('any transcript 1', 0.99, None),
        ],
        base.DetailedTranscript('any transcript 1', 0.99, None)
    ),
    (
        [
            base.DetailedTranscript('any transcript 1', 0.99, None),
            base.DetailedTranscript('any transcript 2', 0.89, None),
        ],
        base.DetailedTranscript('any transcript 1', 0.99, None)
    ),
    (
        [
            base.DetailedTranscript('any transcript 2', -0.89, None),
            base.DetailedTranscript('any transcript 1', -0.99, None),
        ],
        base.DetailedTranscript('any transcript 2', -0.89, None)
    ),
    (
        [
            base.DetailedTranscript('any transcript 2', 0.89, None),
            base.DetailedTranscript('any transcript 1', 0.99, None),
        ],
        base.DetailedTranscript('any transcript 1', 0.99, None)
    ),
])
def test_detailed_transcripts_best_transcript(transcripts, expected_best_transcript):
    subject = base.DetailedTranscripts(transcripts)

    actual_best_transcript = subject.best_transcript

    assert expected_best_transcript == actual_best_transcript


def test_base_speech_to_text_getting_best_transcribed_audio():
    any_audio = AudioSample.from_silence(1, 44100)
    subject = base.BaseSpeechToText()
    subject.transcribe_audio_detailed = mock.MagicMock(return_value=base.DetailedTranscripts([
        base.DetailedTranscript('any transcript 1', 0.99, None),
        base.DetailedTranscript('any transcript 2', 0.89, None),
    ]))

    actual_transcript = subject.transcribe_audio(any_audio)

    assert 'any transcript 1' == actual_transcript
    subject.transcribe_audio_detailed.assert_called_once_with(any_audio, n_transcriptions=1)
