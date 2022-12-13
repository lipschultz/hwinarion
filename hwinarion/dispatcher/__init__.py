from enum import Enum, auto
from typing import Generator, List

from loguru import logger

from hwinarion.listeners.base import BackgroundListener
from hwinarion.speech_to_text import BaseSpeechToText


class ActResult(Enum):
    TEXT_PROCESSED = auto()
    TEXT_NOT_PROCESSED = auto()
    PROCESS_FUTURE_TEXT = auto()


class BaseAction:
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled

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

    def run(self) -> None:
        self.start_listening()

        try:
            for text in self._get_transcribed_text():
                logger.debug(f"Got text (len={len(text)}): {text}")
                for action in self.actions:
                    result = action.act(text)
                    if result == ActResult.TEXT_PROCESSED:
                        logger.info(f"{action} consumed {text}")
                        break
                else:
                    # No action consumed the text
                    logger.info(f"Unconsumed text: {text}")
        finally:
            self.stop_listening()
