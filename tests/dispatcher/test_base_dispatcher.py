from unittest import mock

from hwinarion.dispatcher import BaseAction, BaseDispatcher


class TestBaseDispatcher:
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
