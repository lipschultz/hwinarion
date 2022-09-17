class BaseSpeechToText:
    def __init__(self):
        pass

    def transcribe_audio(self, audio) -> str:
        raise NotImplementedError
