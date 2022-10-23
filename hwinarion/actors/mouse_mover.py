"""
isort:skip_file
"""

import math
import multiprocessing
import platform
import queue
import sys
from typing import Callable, Union, Tuple, Iterator

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
VelocityType = Union[float, int]
TweenType = Callable[[Union[int, float]], float]


class BaseAction:
    def setup(self) -> None:
        pass

    def step(self) -> Iterator[float]:
        raise NotImplementedError


class MouseMoveAction(BaseAction):
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


class MouseToAction(MouseMoveAction):
    def __init__(self, to_position: PositionType, velocity: Union[float, int]):
        super().__init__(velocity)
        self.to_position = (
            to_position if isinstance(to_position, pyautogui.Point) else pyautogui.Point(to_position[0], to_position[1])
        )


class MouseLeftAction(MouseMoveAction):
    def __init__(self, velocity: VelocityType):
        super().__init__(velocity)

    def setup(self) -> None:
        x = 0
        y = pyautogui.position().y
        self.to_position = pyautogui.Point(x, y)
        super().setup()


class MouseUpAction(MouseMoveAction):
    def __init__(self, velocity: VelocityType):
        super().__init__(velocity)

    def setup(self) -> None:
        x = pyautogui.position().x
        y = 0
        self.to_position = pyautogui.Point(x, y)
        super().setup()


class MouseRightAction(MouseMoveAction):
    def __init__(self, velocity: VelocityType):
        super().__init__(velocity)

    def setup(self) -> None:
        x = pyautogui.size().width
        y = pyautogui.position().y
        self.to_position = pyautogui.Point(x, y)
        super().setup()


class MouseDownAction(MouseMoveAction):
    def __init__(self, velocity: VelocityType):
        super().__init__(velocity)

    def setup(self) -> None:
        x = pyautogui.position().x
        y = pyautogui.size().height
        self.to_position = pyautogui.Point(x, y)
        super().setup()


class MouseStopAction(BaseAction):
    def step(self) -> Iterator[float]:
        return iter([])


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
        logger.info(f"Starting actor")
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
        logger.info(f"Stopping actor")
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

            if isinstance(requested_action, MouseStopAction):
                if isinstance(action, MouseMoveAction):
                    action = None
                    sleep_time = None
            elif isinstance(requested_action, MouseMoveAction):
                action = requested_action

            if action is not None:
                logger.info(f"Taking action: {action}")
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
        self.queue.put(MouseToAction((x, y), velocity))

    def move_left(self, velocity: float = SPEED_NORMAL) -> None:
        self.queue.put(MouseLeftAction(velocity))

    def move_right(self, velocity: float = SPEED_NORMAL) -> None:
        self.queue.put(MouseRightAction(velocity))

    def move_up(self, velocity: float = SPEED_NORMAL) -> None:
        self.queue.put(MouseUpAction(velocity))

    def move_down(self, velocity: float = SPEED_NORMAL) -> None:
        self.queue.put(MouseDownAction(velocity))

    def stop_moving(self):
        self.queue.put(MouseStopAction())
