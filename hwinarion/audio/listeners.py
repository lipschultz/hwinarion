import threading
from dataclasses import dataclass
from enum import Enum
from queue import SimpleQueue
from typing import List, Callable, Optional

from hwinarion.audio.base import AudioSample, BaseAudioSource, TimeType

FrameStateEnum = Enum('FrameStateEnum', ('LISTEN', 'PAUSE', 'STOP'))


@dataclass
class AnnotatedFrame:
    frame: AudioSample
    state: FrameStateEnum


class ListenerRunningError(Exception):
    pass


class BackgroundListener:
    def __init__(self, listener: 'BaseListener', listener_kwargs: dict):
        self._listener = listener
        self._listener_kwargs = listener_kwargs
        self._thread = None
        self.__is_running = False
        self.queue = SimpleQueue()

    def start(self) -> None:
        if self._thread is not None:
            raise ListenerRunningError()
        self._thread = threading.Thread(target=self._listen)
        self._thread.daemon = True
        self.__is_running = True
        self._thread.start()

    def stop(self, timeout=None) -> bool:
        """
        Returns True if the listening thread stopped, False otherwise.
        """
        self.__is_running = False
        self._thread.join(timeout)
        return_value = timeout is None or not self._thread.is_alive()
        self._thread = None
        return return_value

    @property
    def is_listening(self) -> bool:
        return self.__is_running

    def _listen(self):
        while self.is_listening:
            try:
                audio = self._listener.listen(**self._listener_kwargs)
                self.queue.put(audio)
                print(audio)
            except Exception as ex:
                pass


class BaseListener:
    def __init__(self, source: BaseAudioSource):
        self.source = source

    def _determine_frame_state(self, latest_frame: AudioSample, all_frames: List[AnnotatedFrame]) -> FrameStateEnum:
        raise NotImplementedError

    def _pre_process_individual_audio_sample(self, audio: AudioSample) -> AudioSample:
        return audio

    def _filter_audio_samples(self, all_frames: List[AnnotatedFrame]) -> List[AnnotatedFrame]:
        previous_state = None
        saved_frames = []
        for frame in all_frames:
            if frame.state == FrameStateEnum.LISTEN or previous_state == FrameStateEnum.LISTEN:
                saved_frames.append(frame)
            previous_state = frame.state
        return saved_frames

    def _join_audio_samples(self, all_frames: List[AnnotatedFrame]) -> AudioSample:
        return AudioSample.from_iterable(frame.frame for frame in all_frames)

    def _post_process_final_audio_sample(self, audio: AudioSample) -> AudioSample:
        return audio

    def listen_frames(self, chunk_size: int = 2**14) -> List[AnnotatedFrame]:
        all_frames = []

        frame = self.source.read(chunk_size)
        frame = self._pre_process_individual_audio_sample(frame)
        frame_state = self._determine_frame_state(frame, all_frames)

        while frame_state != FrameStateEnum.STOP:
            all_frames.append(AnnotatedFrame(frame, frame_state))
            frame = self.source.read(chunk_size)
            frame = self._pre_process_individual_audio_sample(frame)
            frame_state = self._determine_frame_state(frame, all_frames)

        all_frames.append(AnnotatedFrame(frame, frame_state))
        return all_frames

    def listen(self, chunk_size: int = 2**14) -> AudioSample:
        """
        If the end of the source is reached, then listening will end regardless of what ``_determine_frame_state`` will return.
        """
        all_frames = self.listen_frames(chunk_size)
        all_frames = self._filter_audio_samples(all_frames)
        audio_sample = self._join_audio_samples(all_frames)
        audio_sample = self._post_process_final_audio_sample(audio_sample)

        return audio_sample

    def background_listen(self, chunk_size: int = 2**14) -> BackgroundListener:
        return BackgroundListener(self, listener_kwargs={'chunk_size': chunk_size})


class TimeBasedListener(BaseListener):
    def __init__(self, source: BaseAudioSource, total_duration: TimeType):
        super().__init__(source)
        self.total_duration = total_duration

    def _determine_frame_state(self, latest_frame: AudioSample, all_frames: List[AnnotatedFrame]) -> FrameStateEnum:
        duration = sum(frame.frame.n_seconds for frame in all_frames) + latest_frame.n_seconds
        if duration < self.total_duration and len(latest_frame) > 0:
            return FrameStateEnum.LISTEN
        else:
            return FrameStateEnum.STOP


class SilenceBasedListener(BaseListener):
    def __init__(self, source: BaseAudioSource, silence_threshold_rms: int = 500):
        super().__init__(source)
        self.silence_threshold_rms = silence_threshold_rms

    def _determine_frame_state(self, latest_frame: AudioSample, all_frames: List[AnnotatedFrame]) -> FrameStateEnum:
        if latest_frame.rms > self.silence_threshold_rms:
            print(len(all_frames), latest_frame.rms, 'LISTEN')
            return FrameStateEnum.LISTEN
        else:
            if len(all_frames) == 0 or all_frames[-1].state == FrameStateEnum.PAUSE:
                # Haven't started listening yet
                print(len(all_frames), latest_frame.rms,'PAUSE')
                return FrameStateEnum.PAUSE
            else:
                print(len(all_frames), latest_frame.rms,'STOP')
                return FrameStateEnum.STOP


class ConfigurableListener(BaseListener):
    def __init__(
            self,
            source: BaseAudioSource,
            determine_frame_state: Callable,
            *,
            pre_process_individual_audio_sample: Optional[Callable] = None,
            filter_audio_samples: Optional[Callable] = None,
            join_audio_samples: Optional[Callable] = None,
            post_process_final_audio_sample: Optional[Callable] = None,
    ):
        super().__init__(source)

        self._determine_frame_state_fn = determine_frame_state
        self._pre_process_individual_audio_sample_fn = pre_process_individual_audio_sample
        self._filter_audio_samples_fn = filter_audio_samples
        self._join_audio_samples_fn = join_audio_samples
        self._post_process_final_audio_sample_fn = post_process_final_audio_sample

    def _determine_frame_state(self, latest_frame: AudioSample, all_frames: List[AnnotatedFrame]) -> FrameStateEnum:
        return self._determine_frame_state_fn(latest_frame, all_frames)

    def _pre_process_individual_audio_sample(self, audio: AudioSample) -> AudioSample:
        if self._pre_process_individual_audio_sample_fn is None:
            return super()._pre_process_individual_audio_sample(audio)
        return self._pre_process_individual_audio_sample_fn(audio)

    def _filter_audio_samples(self, all_frames: List[AnnotatedFrame]) -> List[AnnotatedFrame]:
        if self._filter_audio_samples_fn is None:
            return super()._filter_audio_samples(all_frames)
        return self._filter_audio_samples_fn(all_frames)

    def _join_audio_samples(self, all_frames: List[AnnotatedFrame]) -> AudioSample:
        if self._join_audio_samples_fn is None:
            return super()._join_audio_samples(all_frames)
        return self._join_audio_samples_fn(all_frames)

    def _post_process_final_audio_sample(self, audio: AudioSample) -> AudioSample:
        if self._post_process_final_audio_sample_fn is None:
            return super()._post_process_final_audio_sample(audio)
        return self._post_process_final_audio_sample_fn(audio)