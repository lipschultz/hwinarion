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

## Batch Processing
def stt_file(filename):
    w = wave.open(filename, 'r')
    assert w.getframerate() == model.sampleRate()

    frames = w.getnframes()
    buffer = w.readframes(frames)

    # Convert from byte array to 16-bit int array
    data16 = np.frombuffer(buffer, dtype=np.int16)

    return model.stt(data16)


start = time.time()
print(stt_file('audio/8455-210777-0068.wav'))
print(f'--Took: {time.time() - start:0.2f} sec')

start = time.time()
print(stt_file('audio/2830-3980-0043.wav'))
print(f'--Took: {time.time() - start:0.2f} sec')

start = time.time()
print(stt_file('audio/4507-16021-0012.wav'))
print(f'--Took: {time.time() - start:0.2f} sec')



context = model.createStream()

filename = 'audio/8455-210777-0068.wav'
w = wave.open(filename, 'r')
frames = w.getnframes()
buffer = w.readframes(frames)
buffer_len = len(buffer)
offset = 0
batch_size = 16384
text = ''
while offset < buffer_len:
    end_offset = offset + batch_size
    chunk = buffer[offset:end_offset]
    data16 = np.frombuffer(chunk, dtype=np.int16)
    model.feedAudioContent(context, data16)
    text = model.intermediateDecode(context)
    print(text)
    offset = end_offset

print(model.finishStream(context))






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


audio = pyaudio.PyAudio()
stream = audio.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=1024,
    stream_callback=process_audio
)
print('Please start speaking, when done press Ctrl-C ...')
stream.start_stream()

try: 
    while stream.is_active():
        time.sleep(0.1)
except KeyboardInterrupt:
    # PyAudio
    stream.stop_stream()
    stream.close()
    audio.terminate()
    print('Finished recording.')
    # DeepSpeech
    text = model.finishStream(context)
    print('Final text = {}'.format(text))

