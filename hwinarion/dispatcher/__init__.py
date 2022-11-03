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

    def register_action(self, action: BaseAction) -> None:
        self.actions.append(action)

    def _get_transcribed_text(self, listener_kwargs: Optional[dict]) -> Generator[str, None, None]:
        listener_kwargs = listener_kwargs or {}
        bg_listener = self.listener.background_listen(**listener_kwargs)
        bg_listener.start()

        logger.debug("Running transcriber")
        while bg_listener.is_listening or not bg_listener.queue.empty():
            audio = bg_listener.get()
            yield self.speech_to_text.transcribe_audio(audio)

    def run(self, listener_kwargs: Optional[dict] = None) -> None:
        for text in self._get_transcribed_text(listener_kwargs):
            logger.debug(f"Got text: {text}")
            for action in self.actions:
                if action.act(text):
                    logger.info(f"{action} consumed {text}")
                    break
            else:
                # No action consumed the text
                logger.info(f"Unconsumed text: {text}")
