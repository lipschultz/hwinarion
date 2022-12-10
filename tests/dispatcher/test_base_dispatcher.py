import queue
from unittest import mock
from unittest.mock import call

from hwinarion.dispatcher import BaseAction, BaseDispatcher
from hwinarion.listeners.base import BackgroundListener


class TestBaseDispatcherRegisteringAction:
    def test_register_action_adds_action_to_list(self):
        subject = BaseDispatcher(mock.MagicMock(), mock.MagicMock())
        any_action = BaseAction("Any Action")

        subject.register_action(any_action)

        assert len(subject.actions) == 1
        assert subject.actions[0] is any_action

    def test_register_additional_action_adds_action_to_list(self):
        subject = BaseDispatcher(mock.MagicMock(), mock.MagicMock())

        first_action = BaseAction("First Action")
        subject.register_action(first_action)

        second_action = BaseAction("Second Action")
        subject.register_action(second_action)

        assert len(subject.actions) == 2
        assert subject.actions[0] is first_action
        assert subject.actions[1] is second_action


class TestBaseDispatcherListener:
    def test_calling_start_listening_starts_listener(self):
        listener = mock.MagicMock()
        listener.start = mock.MagicMock()
        subject = BaseDispatcher(listener, mock.MagicMock())

        subject.start_listening()

        listener.start.assert_called_once_with()

    def test_calling_stop_listening_for_started_listener_stop_listener(self):
        listener = mock.MagicMock()
        is_listening_mock = mock.PropertyMock(side_effect=[False, True])
        type(listener).is_listening = is_listening_mock
        subject = BaseDispatcher(listener, mock.MagicMock())

        subject.start_listening()
        listener.stop.assert_not_called()

        subject.stop_listening()
        listener.stop.assert_called_with()

        assert listener.stop.call_count == 1

    def test_calling_stop_listening_for_not_started_listener(self):
        listener = mock.MagicMock()
        listener.is_listening = False
        subject = BaseDispatcher(listener, mock.MagicMock())

        subject.stop_listening()

        listener.stop.assert_not_called()

    def test_calling_start_listening_for_started_listener_will_restart_listener(self):
        listener = mock.MagicMock()
        is_listening_mock = mock.PropertyMock(side_effect=[False, True])
        type(listener).is_listening = is_listening_mock
        subject = BaseDispatcher(listener, mock.MagicMock())

        subject.start_listening()
        listener.stop.assert_not_called()
        listener.start.assert_called_with()

        subject.start_listening()
        listener.stop.assert_called_with()
        listener.start.assert_called_with()

        assert listener.stop.call_count == 1
        assert listener.start.call_count == 2

    def test_setting_listener_stop_previous_listener(self):
        first_listener = mock.MagicMock()
        is_listening_mock = mock.PropertyMock(side_effect=[False, True])
        type(first_listener).is_listening = is_listening_mock
        second_listener = mock.MagicMock()
        subject = BaseDispatcher(first_listener, mock.MagicMock())
        subject.start_listening()

        subject.set_listener(second_listener)

        first_listener.stop.assert_called_once_with()
        assert subject.listener is second_listener

    @classmethod
    def create_mock_listener(cls, *return_values, n_listening_true=None):
        listener = mock.MagicMock()

        if n_listening_true is None:
            listening = [True] * len(return_values)
        else:
            listening = [True] * n_listening_true
            if len(return_values) > n_listening_true:
                listening += [False] * (len(return_values) - n_listening_true)
        is_listening_mock = mock.PropertyMock(side_effect=[False] + listening + [False])
        type(listener).is_listening = is_listening_mock

        content_queue = queue.SimpleQueue()
        for value in return_values:
            content_queue.put(value)
        listener.empty = content_queue.empty
        listener.get = content_queue.get

        return listener

    @classmethod
    def create_background_listener(cls, *return_values, raise_eof_at_end=True):
        listener = mock.MagicMock()
        last_side_effect = (EOFError(),) if raise_eof_at_end else ()
        listener.listen = mock.MagicMock(side_effect=return_values + last_side_effect)
        return BackgroundListener(listener, {})

    def test_getting_transcribed_text(self):
        # Arrange
        listener = self.create_background_listener(1)

        speech_to_text = mock.MagicMock()
        speech_to_text.transcribe_audio = mock.MagicMock(return_value="one")

        subject = BaseDispatcher(listener, speech_to_text)
        subject.start_listening()

        # Act
        actual_transcriptions = subject._get_transcribed_text()

        # Assert
        assert list(actual_transcriptions) == ["one"]
        speech_to_text.transcribe_audio.assert_called_once_with(1)

    def test_run_continues_until_speech_to_text_queue_is_empty(self):
        # Arrange
        listener = self.create_mock_listener(1, 2, 3, 4, n_listening_true=2)

        speech_to_text = mock.MagicMock()
        speech_to_text.transcribe_audio = mock.MagicMock(side_effect=("one", "two", "three", "four"))

        subject = BaseDispatcher(listener, speech_to_text)
        subject.start_listening()

        # Act
        actual_transcriptions = subject._get_transcribed_text()

        # Assert
        assert list(actual_transcriptions) == ["one", "two", "three", "four"]
        speech_to_text.transcribe_audio.assert_has_calls([call(1), call(2), call(3), call(4)])
        assert speech_to_text.transcribe_audio.call_count == 4

    def test_empty_text_not_yielded(self):
        # Arrange
        listener = self.create_background_listener(1, 2, 3)

        speech_to_text = mock.MagicMock()
        speech_to_text.transcribe_audio = mock.MagicMock(side_effect=["one", "", "two"])

        subject = BaseDispatcher(listener, speech_to_text)
        subject.start_listening()

        # Act
        actual_transcriptions = subject._get_transcribed_text()

        # Assert
        assert list(actual_transcriptions) == ["one", "two"]
        speech_to_text.transcribe_audio.assert_has_calls([call(1), call(2), call(3)])
        assert speech_to_text.transcribe_audio.call_count == 3

    def test_new_listener_used_when_new_listener_set_and_old_listener_not_empty(self):
        # Arrange
        first_listener = self.create_background_listener(1, 2, 3, 4, raise_eof_at_end=False)
        second_listener = self.create_background_listener(7, 8)

        speech_to_text = mock.MagicMock()
        speech_to_text.transcribe_audio = mock.MagicMock(side_effect=("one", "two", "three", "four"))

        subject = BaseDispatcher(first_listener, speech_to_text)
        subject.start_listening()

        transcriber = subject._get_transcribed_text()
        next(transcriber)

        # Act
        subject.set_listener(second_listener, start_listening=True)

        # Act
        actual_transcriptions = list(transcriber)

        # Assert
        assert actual_transcriptions == ["two", "three"]
        speech_to_text.transcribe_audio.assert_has_calls([call(1), call(7), call(8)])
        assert speech_to_text.transcribe_audio.call_count == 3
