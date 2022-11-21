"""
isort:skip_file
"""

import math
import multiprocessing
import platform
import queue
import re
import sys
from typing import Callable, Iterator, List, Tuple, Union, Optional

from loguru import logger

from hwinarion.dispatcher import BaseAction

# pyautogui.moveTo is too slow, so instead of using the public function, using the lower-level platform module which is
# fast enough.
if sys.platform.startswith("java"):
    # from . import _pyautogui_java as pyautogui_module
    raise NotImplementedError("Jython is not yet supported by PyAutoGUI.")
if sys.platform == "darwin":
    from pyautogui import _pyautogui_osx as pyautogui_module
elif sys.platform == "win32":
    from pyautogui import _pyautogui_win as pyautogui_module
elif platform.system() == "Linux":
    import Xlib.threaded  # pylint: disable=unused-import
    from pyautogui import _pyautogui_x11 as pyautogui_module  # pylint: disable=ungrouped-imports
else:
    raise NotImplementedError(f"Your platform {platform.system()} is not supported by PyAutoGUI.")


# This needs to be imported after Xlib.threaded in Linux, so it's after the imports above
import pyautogui  # pylint: disable=wrong-import-position,wrong-import-order


PositionType = Union[Tuple[int, int], pyautogui.Point]
VelocityType = Union[float, int]
TweenType = Callable[[Union[int, float]], float]
TimeDurationType = Union[float, int]


class BaseRequest:
    def setup(self) -> None:
        pass

    def step(self) -> Iterator[float]:
        raise NotImplementedError


class MouseMoveRequest(BaseRequest):
    def __init__(self, velocity: Union[float, int]):
        self.velocity = velocity

        self.to_position = None
        self.from_position = None
        self.step_sleep_time = None
        self.steps = None

        self._iter = None

    def setup(self) -> None:
        self.from_position = pyautogui.position()
        distance = math.dist(self.from_position, self.to_position)
        duration = distance / self.velocity

        n_steps = max(abs(self.from_position.x - self.to_position.x), abs(self.from_position.y - self.to_position.y))

        self.step_sleep_time = duration / n_steps
        if self.step_sleep_time < pyautogui.MINIMUM_SLEEP:
            # Need to remove steps until the sleep time is at least pyautogui.MINIMUM_SLEEP
            n_steps = math.floor(duration / pyautogui.MINIMUM_SLEEP)
            self.step_sleep_time = pyautogui.MINIMUM_SLEEP

        self.steps = [
            (
                math.ceil(self._point_on_line(self.from_position.x, self.to_position.x, i / n_steps)),
                math.ceil(self._point_on_line(self.from_position.y, self.to_position.y, i / n_steps)),
            )
            for i in range(n_steps)
        ]

        if self.steps[-1] != (self.to_position.x, self.to_position.y):
            self.steps.append((self.to_position.x, self.to_position.y))

    @staticmethod
    def _point_on_line(from_point, to_point, percent_through_line) -> float:
        return from_point + (to_point - from_point) * percent_through_line

    def __iter__(self):
        # Take the first action immediately
        pyautogui_module._moveTo(self.steps[0][0], self.steps[0][1])
        for step_x, step_y in self.steps[1:]:
            yield self.step_sleep_time
            pyautogui_module._moveTo(step_x, step_y)

    def __next__(self):
        if self._iter is None:
            self._iter = iter(self)

        return next(self._iter)

    def step(self) -> Iterator[float]:
        # Take the first action immediately
        pyautogui_module._moveTo(self.steps[0][0], self.steps[0][1])
        for step_x, step_y in self.steps[1:]:
            yield self.step_sleep_time
            pyautogui_module._moveTo(step_x, step_y)


class MouseToRequest(MouseMoveRequest):
    def __init__(self, to_position: PositionType, velocity: Union[float, int]):
        super().__init__(velocity)
        self.to_position = (
            to_position if isinstance(to_position, pyautogui.Point) else pyautogui.Point(to_position[0], to_position[1])
        )


class MouseLeftRequest(MouseMoveRequest):
    def __init__(self, velocity: VelocityType):
        super().__init__(velocity)

    def setup(self) -> None:
        x = 0
        y = pyautogui.position().y
        self.to_position = pyautogui.Point(x, y)
        super().setup()


class MouseUpRequest(MouseMoveRequest):
    def __init__(self, velocity: VelocityType):
        super().__init__(velocity)

    def setup(self) -> None:
        x = pyautogui.position().x
        y = 0
        self.to_position = pyautogui.Point(x, y)
        super().setup()


class MouseRightRequest(MouseMoveRequest):
    def __init__(self, velocity: VelocityType):
        super().__init__(velocity)

    def setup(self) -> None:
        x = pyautogui.size().width
        y = pyautogui.position().y
        self.to_position = pyautogui.Point(x, y)
        super().setup()


class MouseDownRequest(MouseMoveRequest):
    def __init__(self, velocity: VelocityType):
        super().__init__(velocity)

    def setup(self) -> None:
        x = pyautogui.position().x
        y = pyautogui.size().height
        self.to_position = pyautogui.Point(x, y)
        super().setup()


class MouseStopRequest(BaseRequest):
    def step(self) -> Iterator[float]:
        return iter([])


class MouseClickRequest(BaseRequest):
    def __init__(self, button, n_clicks: int):
        assert isinstance(n_clicks, int) and n_clicks > 0

        self.button = button
        self.n_clicks = n_clicks

    def step(self) -> Iterator[float]:
        pyautogui.click(button=self.button, clicks=self.n_clicks)
        return iter([])


"""
It may look like this class can be cleaned up and refactored to work better.  However, Xlib (a library that pyautogui
uses for doing mouse/screen stuff on X11 doesn't work well in threaded or multiprocessing environments.  It only seems
to work well when it's used in only one thread/process.  Therefore, all calls to pyautogui are stuffed into one process.
The unused Xlib.threaded import is also meant to help resolve this problem.  I'm not sure it does, but maybe it helps?
"""


class MouseMover:
    SPEED_VERY_SLOW = 122
    SPEED_SLOW = 183
    SPEED_NORMAL = 244
    SPEED_FAST = 367
    SPEED_VERY_FAST = 734

    def __init__(self):
        self._proc = None
        self.queue = None

    def start(self) -> None:
        """
        Start the actor in a separate process.
        """
        logger.info("Starting actor")
        if self._proc is not None:
            if self._proc.is_alive():
                raise Exception(f"{self.__class__.__name__} is already running")

            self.stop()
        self.queue = multiprocessing.Queue()
        self._proc = multiprocessing.Process(target=self._act)
        self._proc.daemon = True
        self._proc.start()

    def stop(self) -> bool:
        """
        Stop the background actor process.

        Returns ``True`` if the process stopped or ``timeout`` was ``None``, ``False`` otherwise.
        """
        logger.info("Stopping actor")
        self._proc.kill()
        self._proc.join()
        return_value = not self._proc.is_alive()
        self._proc = None
        self.queue.close()
        self.queue = None
        return return_value

    @property
    def is_running(self) -> bool:
        """
        Return whether the actor is running.
        """
        return self._proc is not None and self._proc.is_alive()

    def _act(self):
        """
        The method that listens for actions on the queue and acts on them.
        """

        action = None
        sleep_time = 0
        while True:
            if action is None:
                requested_action = self.queue.get()
            else:
                try:
                    requested_action = self.queue.get(True, sleep_time)
                except queue.Empty:
                    requested_action = None

            if requested_action is not None:
                logger.info(f"Received request: {requested_action}")
                requested_action.setup()

            if isinstance(requested_action, MouseStopRequest):
                if isinstance(action, MouseMoveRequest):
                    action = None
                    sleep_time = None
            elif isinstance(requested_action, MouseMoveRequest):
                action = requested_action

            if action is not None:
                try:
                    sleep_time = next(action)
                except StopIteration:
                    sleep_time = None

                if sleep_time is None:
                    action = None

    def move_to(
        self,
        x: Union[int, float],
        y: Union[int, float],
        velocity: float = SPEED_NORMAL,
    ) -> None:
        self.queue.put(MouseToRequest((x, y), velocity))

    def move_left(self, velocity: float = SPEED_NORMAL) -> None:
        self.queue.put(MouseLeftRequest(velocity))

    def move_right(self, velocity: float = SPEED_NORMAL) -> None:
        self.queue.put(MouseRightRequest(velocity))

    def move_up(self, velocity: float = SPEED_NORMAL) -> None:
        self.queue.put(MouseUpRequest(velocity))

    def move_down(self, velocity: float = SPEED_NORMAL) -> None:
        self.queue.put(MouseDownRequest(velocity))

    def stop_moving(self):
        self.queue.put(MouseStopRequest())

    def click(self, button: str, n_clicks: int):
        self.queue.put(MouseClickRequest(button, n_clicks))


class MouseAction(BaseAction):
    def __init__(self):
        super().__init__("mouse")
        self.mouse_mover = MouseMover()
        self.mouse_mover.start()
        self.speed_mapping = {
            "very slow": MouseMover.SPEED_VERY_SLOW,
            "slow": MouseMover.SPEED_SLOW,
            None: MouseMover.SPEED_NORMAL,
            "fast": MouseMover.SPEED_FAST,
            "very fast": MouseMover.SPEED_VERY_FAST,
        }

    @property
    def recognized_words(self) -> List[str]:
        return [
            "move",
            "mouse",
            "left",
            "right",
            "up",
            "down",
            "very",
            "fast",
            "slow",
            "stop",
            "double",
            "triple",
            "middle",
            "click",
        ]

    def parse_text(self, text: str) -> Optional[Tuple]:
        move_parse = re.fullmatch(
            r"move\s+mouse\s+(?P<direction>left|right|up|down)(\s+(?P<speed>(?:very\s+)?(?:fast|slow)))?",
            text,
            flags=re.IGNORECASE,
        )
        if move_parse:
            speed = move_parse["speed"].lower() if move_parse["speed"] else None
            return "move", move_parse["direction"].lower(), speed

        if re.fullmatch(r"stop\s+mouse", text, flags=re.IGNORECASE):
            return ("stop",)

        click_parse = re.fullmatch(
            r"(?P<n_clicks>double|triple)?\s*(?:(?P<button>left|right|middle)[- ]mouse)?\s*click",
            text,
            flags=re.IGNORECASE,
        )
        if click_parse:
            button = click_parse["button"].lower() if click_parse["button"] else "left"
            if click_parse["n_clicks"] is None:
                n_clicks = 1
            elif click_parse["n_clicks"] == "double":
                n_clicks = 2
            elif click_parse["n_clicks"] == "triple":
                n_clicks = 3
            else:
                n_clicks = 1
            return "click", button, n_clicks

        return None

    def act(self, text: str) -> bool:
        parsed_text = self.parse_text(text)
        if parsed_text:
            action, *parameters = parsed_text

            if action == "stop":
                self.mouse_mover.stop_moving()
                return True

            if action == "move":
                direction, speed = parameters
                speed = self.speed_mapping[speed]
                if direction == "left":
                    move_function = self.mouse_mover.move_left
                elif direction == "right":
                    move_function = self.mouse_mover.move_right
                elif direction == "down":
                    move_function = self.mouse_mover.move_down
                elif direction == "up":
                    move_function = self.mouse_mover.move_up
                else:
                    raise KeyError(f"Unrecognized direction: {direction}")

                move_function(speed)
                return True

            if action == "click":
                button, n_clicks = parameters
                self.mouse_mover.click(button, n_clicks)
                return True

        return False
