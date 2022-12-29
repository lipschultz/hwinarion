import enum
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from hwinarion.dispatcher import ActionResult, ActProcessResult, BaseAction, BaseDispatcher


class RecordingState(enum.Enum):
    NOT_RECORDING = enum.auto()
    RECORDING = enum.auto()


class ActionRecorderAction(BaseAction):
    def __init__(self, dispatcher: BaseDispatcher, save_location: Union[str, Path]):
        super().__init__("ActionRecorder")
        self._dispatcher = dispatcher
        self._save_location = Path(save_location)
        self._state = RecordingState.NOT_RECORDING  # type: RecordingState
        self._recorded_actions = None  # type: Optional[list]

    @property
    def recognized_words(self) -> List[str]:
        return ["record", "action", "stop", "recording"]  # pragma: no cover

    @property
    def recording_state(self) -> RecordingState:
        return self._state

    @property
    def is_recording(self) -> bool:
        return self.recording_state is RecordingState.RECORDING

    def _perform_action(self, text) -> None:
        acted_action, result = self._dispatcher._act_on_text(text, [self], get_recording_data=True)
        self._recorded_actions.append(
            {
                "time": datetime.now().isoformat(),
                "text": text,
                "action": acted_action,
                "process_result": result.process_result,
                "recording_data": result.recording_data,
            }
        )

    def _json_encoder(self, value):
        if isinstance(value, BaseAction):
            return value.name
        elif isinstance(value, ActProcessResult):
            return value.name
        elif isinstance(value, np.ndarray):
            return value.tolist()
        else:
            raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")  # pragma: no cover

    def _save_recording(self) -> None:
        with open(self._save_location / (datetime.now().isoformat() + ".json"), "w") as fp:
            json.dump(self._recorded_actions, fp, default=self._json_encoder)
        self._recorded_actions = []

    def act(self, text: str, *, get_recording_data: bool = False) -> ActionResult:
        if not self.enabled:
            return ActionResult(ActProcessResult.TEXT_NOT_PROCESSED)

        if self._state == RecordingState.RECORDING and text.lower() == "stop recording":
            self._save_recording()
            self._state = RecordingState.NOT_RECORDING
            return ActionResult(ActProcessResult.TEXT_PROCESSED)
        elif self._state == RecordingState.RECORDING:
            self._perform_action(text)
            return ActionResult(ActProcessResult.PROCESS_FUTURE_TEXT)
        elif text.lower() == "record action":
            self._state = RecordingState.RECORDING
            self._recorded_actions = []
            return ActionResult(ActProcessResult.PROCESS_FUTURE_TEXT)
        else:
            return ActionResult(ActProcessResult.TEXT_NOT_PROCESSED)
