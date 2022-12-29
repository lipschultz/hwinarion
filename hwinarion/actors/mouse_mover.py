import re
from typing import List, Optional, Tuple

import numpy as np

from hwinarion.dispatcher import ActionResult, ActProcessResult, BaseAction
from hwinarion.screen.pyautogui_wrapper import InterruptibleScreenInteractor


class MouseAction(BaseAction):
    def __init__(self):
        super().__init__("mouse")
        self.screen_interactor = InterruptibleScreenInteractor()
        self.screen_interactor.start()
        self.speed_mapping = {
            "very slow": InterruptibleScreenInteractor.SPEED_VERY_SLOW,
            "slow": InterruptibleScreenInteractor.SPEED_SLOW,
            None: InterruptibleScreenInteractor.SPEED_NORMAL,
            "fast": InterruptibleScreenInteractor.SPEED_FAST,
            "very fast": InterruptibleScreenInteractor.SPEED_VERY_FAST,
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
        ]  # pragma: no cover

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
            r"(?P<n_clicks>single|double|triple)?\s*(?:(?P<button>left|right|middle)[- ])?mouse\s*click",
            text,
            flags=re.IGNORECASE,
        )
        if click_parse:
            button = click_parse["button"].lower() if click_parse["button"] else "left"
            if click_parse["n_clicks"] in (None, "single"):
                n_clicks = 1
            elif click_parse["n_clicks"] == "double":
                n_clicks = 2
            elif click_parse["n_clicks"] == "triple":
                n_clicks = 3
            else:
                raise ValueError(f"Unrecognized number of clicks: {click_parse['n_clicks']}")  # pragma: no cover
            return "click", button, n_clicks

        return None

    def act(self, text: str, *, get_recording_data: bool = False) -> ActionResult:
        if not self.enabled:
            return ActionResult(ActProcessResult.TEXT_NOT_PROCESSED)

        recording_data = {}

        parsed_text = self.parse_text(text)
        if parsed_text:
            action, *parameters = parsed_text
            recording_data["action"] = action
            recording_data["parameters"] = parameters
            if get_recording_data:
                recording_data["mouse_position"] = tuple(self.screen_interactor.get_mouse_position())
                recording_data["screen"] = np.asarray(self.screen_interactor.get_screenshot().tobytes())

            if action == "stop":
                self.screen_interactor.stop_moving()
                return ActionResult(ActProcessResult.TEXT_PROCESSED, recording_data)

            if action == "move":
                direction, speed = parameters
                speed = self.speed_mapping[speed]
                recording_data["speed"] = speed
                if direction == "left":
                    move_function = self.screen_interactor.move_left
                elif direction == "right":
                    move_function = self.screen_interactor.move_right
                elif direction == "down":
                    move_function = self.screen_interactor.move_down
                elif direction == "up":
                    move_function = self.screen_interactor.move_up
                else:
                    raise KeyError(f"Unrecognized direction: {direction}")  # pragma: no cover

                move_function(speed)
                return ActionResult(ActProcessResult.TEXT_PROCESSED, recording_data)

            if action == "click":
                button, n_clicks = parameters
                self.screen_interactor.click(button, n_clicks)
                return ActionResult(ActProcessResult.TEXT_PROCESSED, recording_data)

        return ActionResult(ActProcessResult.TEXT_NOT_PROCESSED)
