from typing import Generator, List

from loguru import logger

from hwinarion.listeners.base import BackgroundListener
from hwinarion.speech_to_text import BaseSpeechToText


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

    def act(self, text: str) -> bool:
        """
        Return True if the action acted on the text, False otherwise.
        """
        raise NotImplementedError  # pragma: no cover


class BaseDispatcher:
    def __init__(self, listener: BackgroundListener, speech_to_text: BaseSpeechToText):
        self.listener = listener
        self.speech_to_text = speech_to_text
        self.actions = []  # type: List[BaseAction]

    def register_action(self, action: BaseAction) -> None:
        self.actions.append(action)

    def set_listener(self, listener: BackgroundListener) -> None:
        self.stop_listening()
        self.listener = listener

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
            logger.debug("Heard audio:", audio)
            transcribed_text = self.speech_to_text.transcribe_audio(audio)
            if len(transcribed_text) > 0:
                yield transcribed_text

    def run(self) -> None:
        self.start_listening()

        try:
            for text in self._get_transcribed_text():
                logger.debug(f"Got text (len={len(text)}): {text}")
                if text.lower() == "stop":
                    logger.info("Stopping dispatch")
                    break
                for action in self.actions:
                    if action.act(text):
                        logger.info(f"{action} consumed {text}")
                        break
                else:
                    # No action consumed the text
                    logger.info(f"Unconsumed text: {text}")
        finally:
            self.stop_listening()
