from typing import Generator, List, Optional

from loguru import logger

from hwinarion.listeners.base import BaseListener
from hwinarion.speech_to_text import BaseSpeechToText


class BaseAction:
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled

    def act(self, text: str) -> bool:
        """
        Return True if the action acted on the text, False otherwise
        """
        raise NotImplementedError


class BaseDispatcher:
    def __init__(self, listener: BaseListener, speech_to_text: BaseSpeechToText):
        self.listener = listener
        self.speech_to_text = speech_to_text
        self.actions = []  # type: List[BaseAction]
        self._bg_listener = None

    def register_action(self, action: BaseAction) -> None:
        self.actions.append(action)

    def _get_transcribed_text(self, listener_kwargs: Optional[dict]) -> Generator[str, None, None]:
        listener_kwargs = listener_kwargs or {}
        self._bg_listener = self.listener.background_listen(**listener_kwargs)
        self._bg_listener.start()

        logger.debug("Running transcriber")
        while self._bg_listener.is_listening or not self._bg_listener.queue.empty():
            audio = self._bg_listener.get()
            transcribed_text = self.speech_to_text.transcribe_audio(audio)
            if len(transcribed_text) > 0:
                yield transcribed_text

    def stop(self) -> None:
        self._bg_listener.stop()

    def run(self, listener_kwargs: Optional[dict] = None) -> None:
        try:
            for text in self._get_transcribed_text(listener_kwargs):
                logger.debug(f"Got text (len={len(text)}): {text}")
                for action in self.actions:
                    if action.act(text):
                        logger.info(f"{action} consumed {text}")
                        break
                else:
                    # No action consumed the text
                    logger.info(f"Unconsumed text: {text}")
        finally:
            self.stop()
