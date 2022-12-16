from collections.abc import Container
from enum import Enum, auto
from typing import Generator, List, Optional, Tuple

from loguru import logger

from hwinarion.listeners.base import BackgroundListener
from hwinarion.speech_to_text.base import BaseSpeechToText


class ActResult(Enum):
    TEXT_PROCESSED = auto()
    TEXT_NOT_PROCESSED = auto()
    PROCESS_FUTURE_TEXT = auto()


class BaseAction:
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

    @property
    def recognized_words(self) -> List[str]:
        """
        Return a list of words the action uses.  This property may be used to customize the language model used by the
        speech-to-text engine.
        """
        raise NotImplementedError  # pragma: no cover

    def act(self, text: str) -> ActResult:
        """
        Return True if the action acted on the text, False otherwise.
        Determine whether the action should act on the text and act if appropriate.

        Returns ``ActResult.TEXT_PROCESSED`` if the action acted on the text and no further processing by other actions
        should take place.

        Returns ``ActResult.TEXT_NOT_PROCESSED`` if another action should act on the text.

        Returns ``ActResult.PROCESS_FUTURE_TEXT`` if the action acted on the text, no further processing by other
        actions should take place, and this action should be the first to receive the next text.
        """
        raise NotImplementedError  # pragma: no cover


class BaseDispatcher:
    def __init__(self, listener: BackgroundListener, speech_to_text: BaseSpeechToText):
        self.listener = listener
        self.speech_to_text = speech_to_text
        self.actions = []  # type: List[BaseAction]

    def register_action(self, action: BaseAction) -> None:
        self.actions.append(action)

    def set_listener(self, listener: BackgroundListener, start_listening=False) -> None:
        self.stop_listening()
        self.listener = listener
        if start_listening:
            self.listener.start()

    def start_listening(self) -> None:
        self.stop_listening()
        self.listener.start()

    def stop_listening(self) -> None:
        if self.listener.is_listening:
            self.listener.stop()

    def _get_transcribed_text(self) -> Generator[str, None, None]:
        logger.debug("Running transcriber")
        while self.listener.is_listening or not self.listener.empty():
            audio = self.listener.get()
            logger.debug(f"Heard audio: {audio}")
            transcribed_text = self.speech_to_text.transcribe_audio(audio)
            if len(transcribed_text) > 0:
                yield transcribed_text

    def _act_on_text(
        self, text: str, skip_actions: Optional[Container[BaseAction]] = None
    ) -> Tuple[Optional[BaseAction], ActResult]:
        """
        Loops over the actions (skipping any listed in ``skip_actions`` until one acts on the provided text.  Returns
        a tuple of the action that acted on the text and what its ActResult was.  If no action acted on the text, then
        returns ``(None, ActResult.TEXT_NOT_PROCESSED)``
        """
        skip_actions = skip_actions or ()

        for action in self.actions:
            if action in skip_actions:
                continue

            logger.info(f"considering action: {action}")
            result = action.act(text)
            if result == ActResult.TEXT_PROCESSED:
                logger.info(f"{action} consumed {text!r}")
                return action, result
            elif result == ActResult.PROCESS_FUTURE_TEXT:
                logger.info(f"{action} consumed {text!r} and will process future text")
                return action, result

        # No action consumed the text
        logger.info(f"Unconsumed text: {text}")
        return None, ActResult.TEXT_NOT_PROCESSED

    def run(self) -> None:
        self.start_listening()

        try:
            action_to_consider_first = None  # type: Optional[BaseAction]
            for text in self._get_transcribed_text():
                logger.debug(f"Got text (len={len(text)}): {text}")

                if action_to_consider_first is not None:
                    logger.debug(f"Trying first: {action_to_consider_first}")
                    action_result = action_to_consider_first.act(text)
                    if action_result == ActResult.TEXT_PROCESSED:
                        action_to_consider_first = None
                    elif action_result == ActResult.TEXT_NOT_PROCESSED:
                        acted_action, action_result = self._act_on_text(text, [action_to_consider_first])
                        action_to_consider_first = acted_action
                else:
                    acted_action, action_result = self._act_on_text(text)
                    if action_result == ActResult.PROCESS_FUTURE_TEXT:
                        action_to_consider_first = acted_action

                if action_result == ActResult.TEXT_NOT_PROCESSED:
                    # No action consumed the text
                    logger.info(f"Unconsumed text: {text}")
        finally:
            self.stop_listening()
