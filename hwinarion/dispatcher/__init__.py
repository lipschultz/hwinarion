from typing import Generator, Optional

from hwinarion.listeners.base import BaseListener
from hwinarion.speech_to_text import BaseSpeechToText


class BaseAction:
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled

    def act(self, text: str) -> bool:
        pass


class BaseDispatcher:
    def __init__(self, listener: BaseListener, speech_to_text: BaseSpeechToText):
        self.listener = listener
        self.speech_to_text = speech_to_text
        self.actions = []

    def run(self, listener_kwargs: Optional[dict]) -> Generator[str, None, None]:
        listener_kwargs = listener_kwargs or {}
        bg_listener = self.listener.background_listen(**listener_kwargs)

        while audio := bg_listener.queue.get():
            yield self.speech_to_text.transcribe_audio(audio)
