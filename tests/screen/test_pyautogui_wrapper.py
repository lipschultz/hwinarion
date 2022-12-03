import math
from unittest import mock

import pyautogui

from hwinarion.screen.pyautogui_wrapper import MouseMoveRequest


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
