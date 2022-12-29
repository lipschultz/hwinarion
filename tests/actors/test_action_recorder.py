import json
from datetime import datetime
from unittest import mock

import numpy as np
from freezegun import freeze_time

from hwinarion.actors.action_recorder import ActionRecorderAction
from hwinarion.dispatcher import ActionResult, ActProcessResult, BaseAction, BaseDispatcher


class TestActionRecorder:
    def test_act_returns_not_processed_when_disabled(self, tmp_path):
        dispatcher = mock.MagicMock()
        actor = ActionRecorderAction(dispatcher, tmp_path)
        actor.enabled = False

        result = actor.act("record action")

        assert isinstance(result, ActionResult)
        assert result.process_result is ActProcessResult.TEXT_NOT_PROCESSED
        assert not actor.is_recording

    def test_starting_record_action(self, tmp_path):
        dispatcher = mock.MagicMock()
        actor = ActionRecorderAction(dispatcher, tmp_path)

        result = actor.act("record action")

        assert isinstance(result, ActionResult)
        assert result.process_result is ActProcessResult.PROCESS_FUTURE_TEXT

        assert actor.is_recording

    def test_calling_subaction_when_in_record_mode(self, tmp_path):
        # Arrange
        acted_subaction = mock.MagicMock()
        acted_subaction.act = mock.MagicMock(return_value=ActionResult(ActProcessResult.TEXT_PROCESSED))
        dispatcher = BaseDispatcher(mock.MagicMock(), mock.MagicMock())
        dispatcher.register_action(acted_subaction)
        actor = ActionRecorderAction(dispatcher, tmp_path)
        actor.act("record action")

        # Act
        result = actor.act("do subaction")

        # Assert
        assert isinstance(result, ActionResult)
        assert result.process_result is ActProcessResult.PROCESS_FUTURE_TEXT

        acted_subaction.act.assert_called_once_with("do subaction", get_recording_data=True)

        assert actor.is_recording

    @freeze_time("2022-07-13 01:02:03")
    def test_recording_subaction_with_no_extra_recording_data(self, tmp_path):
        # Arrange
        expected_time = datetime.now().isoformat()
        acted_subaction = mock.MagicMock()
        acted_subaction.act = mock.MagicMock(return_value=ActionResult(ActProcessResult.TEXT_PROCESSED))
        dispatcher = BaseDispatcher(mock.MagicMock(), mock.MagicMock())
        dispatcher.register_action(acted_subaction)
        actor = ActionRecorderAction(dispatcher, tmp_path)
        actor.act("record action")

        # Act
        result = actor.act("do subaction")

        # Assert
        assert isinstance(result, ActionResult)
        assert result.process_result is ActProcessResult.PROCESS_FUTURE_TEXT

        acted_subaction.act.assert_called_once_with("do subaction", get_recording_data=True)

        assert len(actor._recorded_actions) == 1
        assert actor._recorded_actions[0] == {
            "time": expected_time,
            "text": "do subaction",
            "action": acted_subaction,
            "process_result": ActProcessResult.TEXT_PROCESSED,
            "recording_data": None,
        }

    @freeze_time("2022-07-13 01:02:03")
    def test_recording_subaction_with_extra_recording_data(self, tmp_path):
        # Arrange
        expected_time = datetime.now().isoformat()
        acted_subaction = mock.MagicMock()
        acted_subaction.act = mock.MagicMock(
            return_value=ActionResult(ActProcessResult.TEXT_PROCESSED, {"number": 1, "text": "any text"})
        )
        dispatcher = BaseDispatcher(mock.MagicMock(), mock.MagicMock())
        dispatcher.register_action(acted_subaction)
        actor = ActionRecorderAction(dispatcher, tmp_path)
        actor.act("record action")

        # Act
        result = actor.act("do subaction")

        # Assert
        assert isinstance(result, ActionResult)
        assert result.process_result is ActProcessResult.PROCESS_FUTURE_TEXT

        acted_subaction.act.assert_called_once_with("do subaction", get_recording_data=True)

        assert len(actor._recorded_actions) == 1
        assert actor._recorded_actions[0] == {
            "time": expected_time,
            "text": "do subaction",
            "action": acted_subaction,
            "process_result": ActProcessResult.TEXT_PROCESSED,
            "recording_data": {"number": 1, "text": "any text"},
        }

    @freeze_time("2022-07-13 01:02:03")
    def test_unrecognized_text_in_record_mode(self, tmp_path):
        # Arrange
        expected_time = datetime.now().isoformat()
        acted_subaction = mock.MagicMock()
        acted_subaction.act = mock.MagicMock(return_value=ActionResult(ActProcessResult.TEXT_NOT_PROCESSED))
        dispatcher = BaseDispatcher(mock.MagicMock(), mock.MagicMock())
        dispatcher.register_action(acted_subaction)
        actor = ActionRecorderAction(dispatcher, tmp_path)
        actor.act("record action")

        # Act
        result = actor.act("do subaction")

        # Assert
        assert isinstance(result, ActionResult)
        assert result.process_result is ActProcessResult.PROCESS_FUTURE_TEXT

        acted_subaction.act.assert_called_once_with("do subaction", get_recording_data=True)

        assert len(actor._recorded_actions) == 1
        assert actor._recorded_actions[0] == {
            "time": expected_time,
            "text": "do subaction",
            "action": None,
            "process_result": ActProcessResult.TEXT_NOT_PROCESSED,
            "recording_data": None,
        }

    def test_stopping_record_mode_when_not_in_record_mode_returns_text_not_processed(self, tmp_path):
        dispatcher = mock.MagicMock()
        actor = ActionRecorderAction(dispatcher, tmp_path)

        result = actor.act("stop recording")

        assert isinstance(result, ActionResult)
        assert result.process_result is ActProcessResult.TEXT_NOT_PROCESSED

        assert not actor.is_recording

    @freeze_time("2022-07-13 01:02:03")
    def test_stopping_record_mode_with_no_recorded_actions(self, tmp_path):
        # Arrange
        expected_time = datetime.now().isoformat()
        dispatcher = BaseDispatcher(mock.MagicMock(), mock.MagicMock())
        actor = ActionRecorderAction(dispatcher, tmp_path)
        actor.act("record action")

        # Act
        result = actor.act("stop recording")

        # Assert
        assert isinstance(result, ActionResult)
        assert result.process_result is ActProcessResult.TEXT_PROCESSED

        assert len(actor._recorded_actions) == 0
        assert not actor.is_recording

        output_file = tmp_path / (expected_time + ".json")
        assert output_file.exists()
        with open(output_file) as fp:
            output_content = json.load(fp)
        assert output_content == []

    @freeze_time("2022-07-13 01:02:03")
    def test_stopping_record_mode_with_recorded_action_and_no_extra_recorded_data(self, tmp_path):
        # Arrange
        expected_time = datetime.now().isoformat()

        class SubAction(BaseAction):
            def __init__(self):
                super().__init__("SubAction")

            def act(self, text: str, *, get_recording_data: bool = False) -> ActionResult:
                return ActionResult(ActProcessResult.TEXT_PROCESSED)

        acted_subaction = SubAction()
        dispatcher = BaseDispatcher(mock.MagicMock(), mock.MagicMock())
        dispatcher.register_action(acted_subaction)
        actor = ActionRecorderAction(dispatcher, tmp_path)
        actor.act("record action")
        actor.act("do subaction")

        # Act
        result = actor.act("stop recording")

        # Assert
        assert isinstance(result, ActionResult)
        assert result.process_result is ActProcessResult.TEXT_PROCESSED

        assert len(actor._recorded_actions) == 0
        assert not actor.is_recording

        output_file = tmp_path / (expected_time + ".json")
        assert output_file.exists()
        with open(output_file) as fp:
            output_content = json.load(fp)
        assert output_content == [
            {
                "time": expected_time,
                "text": "do subaction",
                "action": "SubAction",
                "process_result": "TEXT_PROCESSED",
                "recording_data": None,
            }
        ]

    @freeze_time("2022-07-13 01:02:03")
    def test_stopping_record_mode_with_recorded_action_and_simple_recorded_data(self, tmp_path):
        # Arrange
        expected_time = datetime.now().isoformat()

        class SubAction(BaseAction):
            def __init__(self):
                super().__init__("SubAction")

            def act(self, text: str, *, get_recording_data: bool = False) -> ActionResult:
                return ActionResult(
                    ActProcessResult.TEXT_PROCESSED, [{"text": "asdf", "int": 1, "float": 13.7, "none": None}]
                )

        acted_subaction = SubAction()
        dispatcher = BaseDispatcher(mock.MagicMock(), mock.MagicMock())
        dispatcher.register_action(acted_subaction)
        actor = ActionRecorderAction(dispatcher, tmp_path)
        actor.act("record action")
        actor.act("do subaction")

        # Act
        result = actor.act("stop recording")

        # Assert
        assert isinstance(result, ActionResult)
        assert result.process_result is ActProcessResult.TEXT_PROCESSED

        assert len(actor._recorded_actions) == 0
        assert not actor.is_recording

        output_file = tmp_path / (expected_time + ".json")
        assert output_file.exists()
        with open(output_file) as fp:
            output_content = json.load(fp)
        assert output_content == [
            {
                "time": expected_time,
                "text": "do subaction",
                "action": "SubAction",
                "process_result": "TEXT_PROCESSED",
                "recording_data": [{"text": "asdf", "int": 1, "float": 13.7, "none": None}],
            }
        ]

    @freeze_time("2022-07-13 01:02:03")
    def test_stopping_record_mode_with_recorded_action_and_numpy_recorded_data(self, tmp_path):
        # Arrange
        expected_time = datetime.now().isoformat()

        class SubAction(BaseAction):
            def __init__(self):
                super().__init__("SubAction")

            def act(self, text: str, *, get_recording_data: bool = False) -> ActionResult:
                return ActionResult(ActProcessResult.TEXT_PROCESSED, np.array([[1, 2, 3], [4, 5, 6]]))

        acted_subaction = SubAction()
        dispatcher = BaseDispatcher(mock.MagicMock(), mock.MagicMock())
        dispatcher.register_action(acted_subaction)
        actor = ActionRecorderAction(dispatcher, tmp_path)
        actor.act("record action")
        actor.act("do subaction")

        # Act
        result = actor.act("stop recording")

        # Assert
        assert isinstance(result, ActionResult)
        assert result.process_result is ActProcessResult.TEXT_PROCESSED

        assert len(actor._recorded_actions) == 0
        assert not actor.is_recording

        output_file = tmp_path / (expected_time + ".json")
        assert output_file.exists()
        with open(output_file) as fp:
            output_content = json.load(fp)
        assert output_content == [
            {
                "time": expected_time,
                "text": "do subaction",
                "action": "SubAction",
                "process_result": "TEXT_PROCESSED",
                "recording_data": [[1, 2, 3], [4, 5, 6]],
            }
        ]
