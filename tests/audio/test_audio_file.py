from pathlib import Path

import pytest
from pydub import AudioSegment

from hwinarion.audio.audio_file import AudioFile
from hwinarion.audio.base import AudioSample

ANY_AUDIO_FILE = Path(__file__).parent.parent / "resources" / "english-one_two_three.wav"
ANY_AUDIO_FILE_N_FRAMES = 121052


@pytest.mark.parametrize("filepath", [ANY_AUDIO_FILE, str(ANY_AUDIO_FILE)])
def test_loading_audio_file(filepath):
    subject = AudioFile(filepath)

    assert subject.frame_rate == 44100
    assert subject.filepath == Path(filepath)


def test_read_full_file():
    subject = AudioFile(ANY_AUDIO_FILE)
    expected_audio = AudioSample(AudioSegment.from_file(ANY_AUDIO_FILE))

    actual_audio = subject.read(None)

    assert actual_audio == expected_audio


@pytest.mark.parametrize(
    "stop_frame",
    [1, 17, ANY_AUDIO_FILE_N_FRAMES, 200_000],
    ids=["just first frame", "middle of file", "last frame", "beyond end of file"],
)
def test_read_first_frames(stop_frame):
    subject = AudioFile(ANY_AUDIO_FILE)
    expected_audio = AudioSample(AudioSegment.from_file(ANY_AUDIO_FILE)).slice_frame(stop_frame)

    actual_audio = subject.read(stop_frame)

    assert actual_audio == expected_audio


@pytest.mark.parametrize(
    "stop_frame",
    [1, 10, ANY_AUDIO_FILE_N_FRAMES - 17, 200_000],
    ids=["just one frame", "some frames", "last frame", "beyond end of file"],
)
def test_read_after_prior_read(stop_frame):
    read_first_n_frames = 17
    subject = AudioFile(ANY_AUDIO_FILE)
    expected_audio = AudioSample(AudioSegment.from_file(ANY_AUDIO_FILE)).slice_frame(
        read_first_n_frames, read_first_n_frames + stop_frame
    )

    _ = subject.read(read_first_n_frames)
    actual_audio = subject.read(stop_frame)

    assert actual_audio == expected_audio


def test_read_after_reaching_end_of_file():
    subject = AudioFile(ANY_AUDIO_FILE)

    _ = subject.read(ANY_AUDIO_FILE_N_FRAMES)
    with pytest.raises(EOFError):
        subject.read(1)


def test_reset_brings_audio_index_back_to_start():
    n_frames = 17
    subject = AudioFile(ANY_AUDIO_FILE)
    expected_audio = subject.read(n_frames)

    subject.reset()
    actual_audio = subject.read(n_frames)

    assert actual_audio == expected_audio
