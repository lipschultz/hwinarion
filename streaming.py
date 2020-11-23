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



text_so_far = ''
def process_audio(in_data, frame_count, time_info, status):
    global text_so_far
    data16 = np.frombuffer(in_data, dtype=np.int16)
    model.feedAudioContent(context, data16)
    text = model.intermediateDecode(context)
    if text != text_so_far:
        print('Interim text = {}'.format(text))
        text_so_far = text
    return (in_data, pyaudio.paContinue)


chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1
fs = 16000  # Record at 44100 samples per second

seconds = 3
filename = "output.wav"

p = pyaudio.PyAudio()  # Create an interface to PortAudio

print('Recording')

stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True,
    stream_callback=process_audio)

stream.start_stream()

try:
    while stream.is_active():
        time.sleep(0.1)
except KeyboardInterrupt:
    # PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()
    print('Finished recording.')
    # DeepSpeech
    text = model.finishStream(context)
    print('Final text = {}'.format(text))

