"""
isort:skip_file
"""

import math
import multiprocessing
import platform
import queue
import sys
from typing import Callable, Union, Tuple

from loguru import logger


# pyautogui.moveTo is too slow, so instead of using the public function, using the lower-level platform module which is
# fast enough.
if sys.platform.startswith("java"):
    # from . import _pyautogui_java as pyautogui_module
    raise NotImplementedError("Jython is not yet supported by PyAutoGUI.")
elif sys.platform == "darwin":
    from pyautogui import _pyautogui_osx as pyautogui_module
elif sys.platform == "win32":
    from pyautogui import _pyautogui_win as pyautogui_module
elif platform.system() == "Linux":
    import Xlib.threaded  # pylint: disable=unused-import
    from pyautogui import _pyautogui_x11 as pyautogui_module
else:
    raise NotImplementedError("Your platform (%s) is not supported by PyAutoGUI." % (platform.system()))


# This needs to be imported after Xlib.threaded in Linux, so it's after the imports above
import pyautogui


PositionType = Union[Tuple[int, int], pyautogui.Point]
TweenType = Callable[[Union[int, float]], float]

"""
It may look like this class can be cleaned up and refactored to work better.  However, Xlib (a library that pyautogui
uses for doing mouse/screen stuff on X11 doesn't work well in threaded or multiprocessing environments.  It only seems
to work well when it's used in only one thread/process.  Therefore, all calls to pyautogui are stuffed into one process.
THe unused Xlib.threaded import is also meant to help resolve this problem.  I'm not sure it does, but maybe it helps?
"""


class MouseMover:
    SPEED_NORMAL = 367
    SPEED_FAST = 734

    def __init__(self):
        self._proc = None
        self.queue = None

    def start(self) -> None:
        """
        Start the actor in a separate process.
        """
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
        while (requested_action := self.queue.get()) is not None:
            action, *args = requested_action
            if action == "to":
                x, y, velocity = args
            else:
                if action == "left":
                    x = 0
                    y = pyautogui.position().y
                elif action == "right":
                    x = pyautogui.size().width
                    y = pyautogui.position().y
                elif action == "up":
                    x = pyautogui.position().x
                    y = 0
                elif action == "down":
                    y = pyautogui.size().height
                    x = pyautogui.position().x
                else:
                    logger.error(f"Unrecognized action: {action}, {args}")
                    continue

                velocity = args

            self._act_move_mouse(x, y, velocity)

    @staticmethod
    def _point_on_line(from_point, to_point, percent_through_line) -> float:
        return from_point + (to_point - from_point) * percent_through_line

    def _act_move_mouse(self, x: int, y: int, velocity: Union[float, int]):
        current_position = pyautogui.position()

        distance = math.dist((x, y), current_position)
        duration = distance / velocity

        n_steps = max(abs(current_position.x - x), abs(current_position.y - y))

        step_sleep_time = duration / n_steps
        if step_sleep_time < pyautogui.MINIMUM_SLEEP:
            # Need to remove steps until the sleep time is at least pyautogui.MINIMUM_SLEEP
            n_steps = math.floor(duration / pyautogui.MINIMUM_SLEEP)
            step_sleep_time = pyautogui.MINIMUM_SLEEP

        steps = [
            (
                math.ceil(self._point_on_line(current_position.x, x, i / n_steps)),
                math.ceil(self._point_on_line(current_position.y, y, i / n_steps)),
            )
            for i in range(n_steps)
        ]

        if steps[-1] != (x, y):
            steps.append((x, y))

        logger.debug(f"start moving: {current_position} -> ({x}, {y}), {duration:0.5f}, {len(steps)}")
        for i, (step_x, step_y) in enumerate(steps):
            if i != 0:
                try:
                    self.queue.get(True, step_sleep_time)
                except queue.Empty:
                    pass

            pyautogui_module._moveTo(step_x, step_y)
        logger.debug(f"stop moving")

    def move_to(
        self,
        x: Union[int, float],
        y: Union[int, float],
        velocity: float = SPEED_NORMAL,
    ) -> None:
        self.queue.put(("to", x, y, velocity))

    def move_left(self, velocity: float = SPEED_NORMAL) -> None:
        self.queue.put(("left", velocity))

    def move_right(self, velocity: float = SPEED_NORMAL) -> None:
        self.queue.put(("right", velocity))

    def move_up(self, velocity: float = SPEED_NORMAL) -> None:
        self.queue.put(("up", velocity))

    def move_down(self, velocity: float = SPEED_NORMAL) -> None:
        self.queue.put(("down", velocity))

    def stop_moving(self):
        pass
