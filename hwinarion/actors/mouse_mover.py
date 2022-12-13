"""
isort:skip_file
"""

import re
from typing import List, Tuple, Optional

from hwinarion.dispatcher import BaseAction, ActResult
from hwinarion.screen.pyautogui_wrapper import InterruptibleScreenInteractor


class MouseAction(BaseAction):
    def __init__(self):
        super().__init__("mouse")
        self.mouse_mover = InterruptibleScreenInteractor()
        self.mouse_mover.start()
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
                raise ValueError(f"Unrecognized number of clicks: {click_parse['n_clicks']}")
            return "click", button, n_clicks

        return None

    def act(self, text: str) -> ActResult:
        if not self.enabled:
            return ActResult.TEXT_NOT_PROCESSED

        parsed_text = self.parse_text(text)
        if parsed_text:
            action, *parameters = parsed_text

            if action == "stop":
                self.mouse_mover.stop_moving()
                return ActResult.TEXT_PROCESSED

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
                return ActResult.TEXT_PROCESSED

            if action == "click":
                button, n_clicks = parameters
                self.mouse_mover.click(button, n_clicks)
                return ActResult.TEXT_PROCESSED

        return ActResult.TEXT_NOT_PROCESSED
