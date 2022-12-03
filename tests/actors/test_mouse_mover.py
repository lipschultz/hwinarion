import math
from unittest import mock

import pyautogui
import pytest

from hwinarion.actors.mouse_mover import MouseAction, MouseMoveRequest


class TestMouseMoveRequest:
    def test_setup_sets_current_mouse_position_as_from(self):
        mouse_position = pyautogui.Point(17, 31)
        pyautogui.position = mock.MagicMock(return_value=mouse_position)
        subject = MouseMoveRequest(19)
        subject.to_position = pyautogui.Point(7, 11)

        subject.setup()

        assert subject.from_position == mouse_position
        pyautogui.position.assert_called_once_with()

    def test_setup_for_reasonable_speed_and_points(self):
        mouse_position = pyautogui.Point(17, 31)
        speed = 19
        goal_position = pyautogui.Point(10, 100)

        pyautogui.position = mock.MagicMock(return_value=mouse_position)
        subject = MouseMoveRequest(speed)
        subject.to_position = goal_position

        expected_duration = math.dist(mouse_position, subject.to_position) / speed
        max_steps = subject.to_position.y - mouse_position.y
        expected_duration_per_step = expected_duration / max_steps

        subject.setup()

        assert subject.step_sleep_time == expected_duration_per_step
        assert subject.step_sleep_time >= pyautogui.MINIMUM_SLEEP

        assert len(subject.steps) > 1
        assert subject.steps[-1] == goal_position

    def test_setup_sets_step_sleep_time_to_min_sleep_time_if_step_sleep_is_too_short(self):
        mouse_position = pyautogui.Point(17, 31)
        speed = 190
        goal_position = pyautogui.Point(10, 100)

        pyautogui.position = mock.MagicMock(return_value=mouse_position)
        subject = MouseMoveRequest(speed)
        subject.to_position = goal_position

        subject.setup()

        assert subject.step_sleep_time == pyautogui.MINIMUM_SLEEP

        assert len(subject.steps) > 1
        assert subject.steps[-1] == goal_position

    def test_setup_when_mouse_just_jumps_to_destination(self):
        speed = 19000  # The speed is so high that the mouse should just jump straight to the destination
        goal_position = pyautogui.Point(10, 100)

        pyautogui.position = mock.MagicMock(return_value=pyautogui.Point(17, 31))
        subject = MouseMoveRequest(speed)
        subject.to_position = goal_position

        subject.setup()

        assert subject.step_sleep_time == pyautogui.MINIMUM_SLEEP

        assert len(subject.steps) >= 1
        assert subject.steps[-1] == goal_position


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
            ("double mouse click", "left", 2),
            ("double left-mouse click", "left", 2),
            ("double left mouse click", "left", 2),
            ("triple mouse click", "left", 3),
            ("triple left-mouse click", "left", 3),
            ("triple left mouse click", "left", 3),
            ("right-mouse click", "right", 1),
            ("right mouse click", "right", 1),
            ("double right-mouse click", "right", 2),
            ("double right mouse click", "right", 2),
            ("triple right-mouse click", "right", 3),
            ("triple right mouse click", "right", 3),
            ("middle-mouse click", "middle", 1),
            ("middle mouse click", "middle", 1),
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
