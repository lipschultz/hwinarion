from unittest import mock

from hwinarion.dispatcher import BaseAction, BaseDispatcher


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
