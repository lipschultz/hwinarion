"""
isort:skip_file
"""

import math
import multiprocessing
from typing import Callable, Union

import Xlib.threaded  # pylint: disable=unused-import
import pyautogui
from loguru import logger


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

    def _act_move_mouse(self, x, y, velocity):
        distance = math.dist((x, y), pyautogui.position())
        duration = distance / velocity
        logger.debug(f"start moving: ({x}, {y}), {duration}")
        pyautogui.moveTo(x, y, duration)
        logger.debug("stop moving")

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
