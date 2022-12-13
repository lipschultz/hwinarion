from hwinarion.actors import StringLiteralActor
from hwinarion.dispatcher import ActResult
from hwinarion.speech_to_text.base import BaseSpeechToText


class HwinArionAction(StringLiteralActor):
    def __init__(self, dispatcher, wakeword_speech_to_text: BaseSpeechToText):
        super().__init__("HwinArionController")
        self._dispatcher = dispatcher
        self._wakeword_speech_to_text = wakeword_speech_to_text
        self._normal_speech_to_text = None

        self.add_action("stop", self.stop_dispatcher)
        self.add_action("go to sleep", self.switch_to_wakeword_detection)
        self.add_action("wake up", self.switch_from_wakeword_detection)
        self.add_action("wake-up", self.switch_from_wakeword_detection)
        self.add_action("wakeup", self.switch_from_wakeword_detection)

    def stop_dispatcher(self, text) -> ActResult:
        self._dispatcher.stop_listening()
        return ActResult.TEXT_PROCESSED

    def switch_to_wakeword_detection(self, text) -> ActResult:
        self._normal_speech_to_text = self._dispatcher.speech_to_text
        self._dispatcher.speech_to_text = self._wakeword_speech_to_text
        return ActResult.TEXT_PROCESSED

    def switch_from_wakeword_detection(self, text) -> ActResult:
        self._dispatcher.speech_to_text = self._normal_speech_to_text
        return ActResult.TEXT_PROCESSED

    def transform_trigger(self, trigger: str) -> str:
        return trigger.lower()
