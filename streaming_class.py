import time
import wave

import deepspeech
import numpy as np
import pyaudio


model_file_path = 'deepspeech-0.6.0-models/output_graph.pbmm'
beam_width = 500
model = deepspeech.Model(model_file_path, beam_width)

lm_file_path = 'deepspeech-0.6.0-models/lm.binary'
trie_file_path = 'deepspeech-0.6.0-models/trie'
lm_alpha = 0.75
lm_beta = 1.85
model.enableDecoderWithLM(lm_file_path, trie_file_path, lm_alpha, lm_beta)

context = model.createStream()


class StreamingInput:
    def __init__(self, model, rate=16000, frames_per_buffer=1024):
        self.__model = model
        self.__rate = rate
        self.__frames_per_buffer = frames_per_buffer

        self.__sample_format = pyaudio.paInt16
        self.__channels = 1

        self.__text = ''
        self.__pyaudio = None
        self.__stream = None

    def start(self):
        self.__pyaudio = pyaudio.PyAudio()
        self.__stream = self.__pyaudio.open(
            format=self.__sample_format,
            channels=self.__channels,
            rate=self.__rate,
            frames_per_buffer=self.__frames_per_buffer,
            input=True,
            stream_callback=self._process_input
        )

    def stop(self):
        self.__stream.stop_stream()
        self.__stream.close()
        self.__pyaudio.terminate()

        self.__stream = None
        self.__pyaudio = None

    def is_active(self):
        return self.__stream and self.__stream.is_active()

    @property
    def text(self):
        return self.__text

    def _process_input(self, in_data, frame_count, time_info, status):
        data16 = np.frombuffer(in_data, dtype=np.int16)
        self.__model.feedAudioContent(context, data16)
        self.__text = model.intermediateDecode(context)

        return (in_data, pyaudio.paContinue)


filename = "output.wav"

print('Recording')

stream = StreamingInput(model)

stream.start()
try:
    text = ''
    while stream.is_active():
        time.sleep(0.1)
        if stream.text != text:
            text = stream.text
            print('Inter text = {}'.format(text))
except KeyboardInterrupt:
    # PyAudio
    stream.stop()
    print('Finished recording.')
    # DeepSpeech
    text = model.finishStream(context)
    print('Final text = {}'.format(text))

