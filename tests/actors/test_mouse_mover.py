from unittest import mock

import numpy as np
import pyautogui
import pytest
from PIL import Image

from hwinarion.actors.mouse_mover import MouseAction
from hwinarion.dispatcher import ActionResult, ActProcessResult


class TestMouseAction:
    @pytest.mark.parametrize(
        "input_text, expected_direction, expected_speed",
        [
            ("move mouse left", "left", None),
            ("move mouse left fast", "left", "fast"),
            ("move mouse left very fast", "left", "very fast"),
            ("move mouse left slow", "left", "slow"),
            ("move mouse left very slow", "left", "very slow"),
            ("move mouse right", "right", None),
            ("move mouse right fast", "right", "fast"),
            ("move mouse right very fast", "right", "very fast"),
            ("move mouse right slow", "right", "slow"),
            ("move mouse right very slow", "right", "very slow"),
            ("move mouse up", "up", None),
            ("move mouse up fast", "up", "fast"),
            ("move mouse up very fast", "up", "very fast"),
            ("move mouse up slow", "up", "slow"),
            ("move mouse up very slow", "up", "very slow"),
            ("move mouse down", "down", None),
            ("move mouse down fast", "down", "fast"),
            ("move mouse down very fast", "down", "very fast"),
            ("move mouse down slow", "down", "slow"),
            ("move mouse down very slow", "down", "very slow"),
        ],
    )
    def test_move_requests(self, input_text, expected_direction, expected_speed):
        actor = MouseAction()

        actual_action, actual_direction, actual_speed = actor.parse_text(input_text)

        assert actual_action == "move"
        assert actual_direction == expected_direction
        assert actual_speed == expected_speed

    @pytest.mark.parametrize(
        "input_text, expected_direction, expected_speed",
        [
            ("MOVE MOUSE LEFT", "left", None),
            ("MOVE MOUSE LEFT FAST", "left", "fast"),
            ("MOVE MOUSE LEFT VERY FAST", "left", "very fast"),
            ("MOVE MOUSE LEFT SLOW", "left", "slow"),
            ("MOVE MOUSE LEFT VERY SLOW", "left", "very slow"),
            ("MOVE MOUSE RIGHT", "right", None),
            ("MOVE MOUSE RIGHT FAST", "right", "fast"),
            ("MOVE MOUSE RIGHT VERY FAST", "right", "very fast"),
            ("MOVE MOUSE RIGHT SLOW", "right", "slow"),
            ("MOVE MOUSE RIGHT VERY SLOW", "right", "very slow"),
            ("MOVE MOUSE UP", "up", None),
            ("MOVE MOUSE UP FAST", "up", "fast"),
            ("MOVE MOUSE UP VERY FAST", "up", "very fast"),
            ("MOVE MOUSE UP SLOW", "up", "slow"),
            ("MOVE MOUSE UP VERY SLOW", "up", "very slow"),
            ("MOVE MOUSE DOWN", "down", None),
            ("MOVE MOUSE DOWN FAST", "down", "fast"),
            ("MOVE MOUSE DOWN VERY FAST", "down", "very fast"),
            ("MOVE MOUSE DOWN SLOW", "down", "slow"),
            ("MOVE MOUSE DOWN VERY SLOW", "down", "very slow"),
        ],
    )
    def test_capitalization_doesnt_matter_for_move(self, input_text, expected_direction, expected_speed):
        actor = MouseAction()

        actual_action, actual_direction, actual_speed = actor.parse_text(input_text)

        assert actual_action == "move"
        assert actual_direction == expected_direction
        assert actual_speed == expected_speed

    def test_stop_requests(self):
        actor = MouseAction()

        actual_action, *actual_parameters = actor.parse_text("stop mouse")

        assert actual_action == "stop"
        assert len(actual_parameters) == 0

    def test_capitalization_doesnt_matter_for_stop(self):
        actor = MouseAction()

        actual_action, *actual_parameters = actor.parse_text("STOP MOUSE")

        assert actual_action == "stop"
        assert len(actual_parameters) == 0

    @pytest.mark.parametrize(
        "input_text, expected_button, expected_n_clicks",
        [
            ("mouse click", "left", 1),
            ("left-mouse click", "left", 1),
            ("left mouse click", "left", 1),
            ("single mouse click", "left", 1),
            ("single left-mouse click", "left", 1),
            ("single left mouse click", "left", 1),
            ("double mouse click", "left", 2),
            ("double left-mouse click", "left", 2),
            ("double left mouse click", "left", 2),
            ("triple mouse click", "left", 3),
            ("triple left-mouse click", "left", 3),
            ("triple left mouse click", "left", 3),
            ("right-mouse click", "right", 1),
            ("right mouse click", "right", 1),
            ("single right-mouse click", "right", 1),
            ("single right mouse click", "right", 1),
            ("double right-mouse click", "right", 2),
            ("double right mouse click", "right", 2),
            ("triple right-mouse click", "right", 3),
            ("triple right mouse click", "right", 3),
            ("middle-mouse click", "middle", 1),
            ("middle mouse click", "middle", 1),
            ("single middle-mouse click", "middle", 1),
            ("single middle mouse click", "middle", 1),
            ("double middle-mouse click", "middle", 2),
            ("double middle mouse click", "middle", 2),
            ("triple middle-mouse click", "middle", 3),
            ("triple middle mouse click", "middle", 3),
        ],
    )
    def test_click_requests(self, input_text, expected_button, expected_n_clicks):
        actor = MouseAction()

        actual_action, actual_button, actual_n_clicks = actor.parse_text(input_text)

        assert actual_action == "click"
        assert actual_button == expected_button
        assert actual_n_clicks == expected_n_clicks

    @pytest.mark.parametrize(
        "input_text",
        [
            "move mouse stop",
            "move mouse very fast",
            "move mouse fast",
            "move mouse left very",
            "move mouse left and extra text after",
            "move mouse",
        ],
    )
    def test_invalid_requests(self, input_text):
        actor = MouseAction()

        result = actor.parse_text(input_text)

        assert result is None

    def test_act_returns_not_processed_when_disabled(self):
        actor = MouseAction()
        actor.enabled = False

        result = actor.act("any text")

        assert isinstance(result, ActionResult)
        assert result.process_result is ActProcessResult.TEXT_NOT_PROCESSED

    def test_if_parsed_text_returns_none_then_act_takes_no_action(self):
        actor = MouseAction()
        actor.parse_text = mock.MagicMock(return_value=None)

        result = actor.act("any text")

        assert isinstance(result, ActionResult)
        assert result.process_result is ActProcessResult.TEXT_NOT_PROCESSED
        actor.parse_text.assert_called_once_with("any text")

    def test_getting_recording_data_for_stop_action(self):
        screen_interactor = mock.MagicMock()
        screen_interactor.get_mouse_position = mock.MagicMock(return_value=pyautogui.Point(11, 29))
        screenshot = Image.new("RGBA", (10, 10), "#00000000")
        screen_interactor.get_screenshot = mock.MagicMock(return_value=screenshot)
        screen_interactor.stop_moving = mock.MagicMock()
        actor = MouseAction()
        actor.screen_interactor.stop()
        actor.screen_interactor = screen_interactor

        result = actor.act("stop mouse", get_recording_data=True)

        assert isinstance(result, ActionResult)
        assert result.process_result is ActProcessResult.TEXT_PROCESSED
        assert result.recording_data == {
            "action": "stop",
            "parameters": [],
            "mouse_position": (11, 29),
            "screen": np.asarray(screenshot.tobytes()),
        }

        screen_interactor.get_mouse_position.assert_called_once_with()
        screen_interactor.get_screenshot.assert_called_once_with()
        screen_interactor.stop_moving.assert_called_once_with()

    def test_getting_recording_data_for_move_action(self):
        screen_interactor = mock.MagicMock()
        screen_interactor.get_mouse_position = mock.MagicMock(return_value=pyautogui.Point(11, 29))
        screenshot = Image.new("RGBA", (10, 10), "#00000000")
        screen_interactor.get_screenshot = mock.MagicMock(return_value=screenshot)
        screen_interactor.move_down = mock.MagicMock()
        actor = MouseAction()
        actor.screen_interactor.stop()
        actor.screen_interactor = screen_interactor

        result = actor.act("move mouse down very fast", get_recording_data=True)

        assert isinstance(result, ActionResult)
        assert result.process_result is ActProcessResult.TEXT_PROCESSED
        assert result.recording_data == {
            "action": "move",
            "parameters": ["down", "very fast"],
            "mouse_position": (11, 29),
            "screen": np.asarray(screenshot.tobytes()),
            "speed": actor.speed_mapping["very fast"],
        }

        screen_interactor.get_mouse_position.assert_called_once_with()
        screen_interactor.get_screenshot.assert_called_once_with()
        screen_interactor.move_down.assert_called_once_with(actor.speed_mapping["very fast"])

    def test_getting_recording_data_for_click_action(self):
        screen_interactor = mock.MagicMock()
        screen_interactor.get_mouse_position = mock.MagicMock(return_value=pyautogui.Point(11, 29))
        screenshot = Image.new("RGBA", (10, 10), "#00000000")
        screen_interactor.get_screenshot = mock.MagicMock(return_value=screenshot)
        screen_interactor.click = mock.MagicMock()
        actor = MouseAction()
        actor.screen_interactor.stop()
        actor.screen_interactor = screen_interactor

        result = actor.act("triple right mouse click", get_recording_data=True)

        assert isinstance(result, ActionResult)
        assert result.process_result is ActProcessResult.TEXT_PROCESSED
        assert result.recording_data == {
            "action": "click",
            "parameters": ["right", 3],
            "mouse_position": (11, 29),
            "screen": np.asarray(screenshot.tobytes()),
        }

        screen_interactor.get_mouse_position.assert_called_once_with()
        screen_interactor.get_screenshot.assert_called_once_with()
        screen_interactor.click.assert_called_once_with("right", 3)
