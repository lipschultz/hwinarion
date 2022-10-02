from unittest import mock

import numpy as np
import pytest

from hwinarion.audio import base


class TestAudioSample:
    def test_from_to_numpy_array(self):
        input_numpy_array = (10_000 * np.sin(np.linspace(0, 4, 1_000_000))).astype("int16")

        subject = base.AudioSample.from_numpy(input_numpy_array, 44100)
        actual_numpy_array = subject.to_numpy()

        assert subject.frame_rate == 44100
        assert subject.n_channels == 1
        assert subject.bit_depth == 16
        assert subject.sample_width == 2
        assert subject.frame_width == 2
        assert subject.n_frames == 1_000_000
        assert subject.n_seconds == 1_000_000 / 44100
        assert (input_numpy_array == actual_numpy_array).all()

    def test_generate_silence(self):
        subject = base.AudioSample.generate_silence(1, 44100)

        assert subject.frame_rate == 44100
        assert subject.n_channels == 1
        assert (subject.to_numpy() == np.zeros(44100)).all()

    def test_equals_operator_is_true_when_same_audio(self):
        subject1 = base.AudioSample.from_numpy((10_000 * np.sin(np.linspace(0, 4, 500_000))).astype("int16"), 44100)
        subject2 = base.AudioSample.from_numpy((10_000 * np.sin(np.linspace(0, 4, 500_000))).astype("int16"), 44100)

        assert subject1 == subject2

    def test_equals_operator_is_false_when_different_audio(self):
        subject1 = base.AudioSample.from_numpy((10_000 * np.sin(np.linspace(0, 4, 500_000))).astype("int16"), 44100)
        subject2 = base.AudioSample.generate_silence(subject1.n_seconds, 44100)

        assert (subject1 == subject2) is False

    def test_equals_operator_is_false_when_same_audio_but_one_is_longer(self):
        subject1 = base.AudioSample.from_numpy((10_000 * np.sin(np.linspace(0, 4, 500_000))).astype("int16"), 44100)
        subject2 = base.AudioSample.from_numpy((10_000 * np.sin(np.linspace(0, 4, 250_000))).astype("int16"), 44100)

        assert (subject1 == subject2) is False

    @pytest.mark.parametrize(
        "stop_frame", [100, 50_000, 100_000], ids=["under frame limit", "at last frame", "after last frame"]
    )
    def test_frame_slice_stop_only(self, stop_frame):
        numpy_sample = (10_000 * np.sin(np.linspace(0, 2, 50_000))).astype("int16")
        subject = base.AudioSample.from_numpy(numpy_sample, 44100)
        expected_sample = numpy_sample[:stop_frame]

        actual_sample = subject.slice_frame(stop_frame).to_numpy()

        assert (actual_sample == expected_sample).all()

    @pytest.mark.parametrize(
        "start_frame, stop_frame",
        [
            (0, 100),
            (0, 50_000),
            (0, 100_000),
            (100, 1_000),
            (100, 50_000),
            (100, 100_000),
        ],
        ids=[
            "start to middle",
            "start to last frame",
            "start to after last frame",
            "middle to middle",
            "middle to last frame",
            "middle to after last frame",
        ],
    )
    def test_frame_slice_start_and_stop(self, start_frame, stop_frame):
        numpy_sample = (10_000 * np.sin(np.linspace(0, 2, 50_000))).astype("int16")
        subject = base.AudioSample.from_numpy(numpy_sample, 44100)
        expected_sample = numpy_sample[start_frame:stop_frame]

        actual_sample = subject.slice_frame(start_frame, stop_frame).to_numpy()

        assert (actual_sample == expected_sample).all()


class TestBaseAudioSource:
    @pytest.mark.parametrize("seconds", [7, 11.12])
    def test_seconds_to_frame(self, seconds):
        class SubjectBaseAudioSource(base.BaseAudioSource):
            @property
            def frame_rate(self) -> int:
                return 44100

        subject = SubjectBaseAudioSource()

        actual_frames = subject.seconds_to_frame(seconds)

        assert actual_frames == int(seconds * 44100)

    @pytest.mark.parametrize("seconds", [7, 11.12])
    def test_read_seconds_with_number_given(self, seconds):
        frame_rate = 44100

        class SubjectBaseAudioSource(base.BaseAudioSource):
            @property
            def frame_rate(self) -> int:
                return frame_rate

        audio_sample = base.AudioSample.from_numpy(
            (10_000 * np.sin(np.linspace(0, 4, 500_000))).astype("int16"), frame_rate
        )

        subject = SubjectBaseAudioSource()
        subject.read = mock.MagicMock(return_value=audio_sample)

        actual_audio = subject.read_seconds(seconds)

        subject.read.assert_called_once_with(int(seconds * frame_rate))
        assert audio_sample == actual_audio
