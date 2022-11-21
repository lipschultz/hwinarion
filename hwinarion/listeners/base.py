import threading
from dataclasses import dataclass
from enum import Enum
from queue import SimpleQueue
from typing import List, Optional

from loguru import logger

from hwinarion.audio.base import AudioSample, BaseAudioSource


class FrameStateEnum(Enum):
    LISTEN = "LISTEN"
    PAUSE = "PAUSE"
    STOP = "STOP"


@dataclass
class AnnotatedFrame:
    frame: AudioSample
    state: FrameStateEnum


class ListenerRunningError(Exception):
    pass


class BackgroundListener:
    def __init__(self, listener: "BaseListener", listener_kwargs: dict):
        self._listener = listener
        self._listener_kwargs = listener_kwargs
        self._thread = None
        self.queue = SimpleQueue()

    def start(self) -> None:
        """
        Start the background listener in a background thread.  Any audio samples recorded will be available in the
        ``queue`` attribute.
        """
        if self._thread is not None:
            if self._thread.is_alive():
                raise ListenerRunningError("Background listener is already running")
            self.stop()
        self._thread = threading.Thread(target=self._listen)
        self._thread.daemon = True
        self._thread.start()

    def stop(self, timeout: Optional[int] = None) -> bool:
        """
        Stop the background listener.

        ``timeout`` indicates how long to wait when joining the background listener thread.  If ``None`` (default),
        then don't wait.

        Returns ``True`` if the listening thread stopped or ``timeout`` was ``None``, ``False`` otherwise.
        """
        self._thread.join(timeout)
        return_value = timeout is None or not self._thread.is_alive()
        self._thread = None
        return return_value

    @property
    def is_listening(self) -> bool:
        """
        Return whether the listener is running.
        """
        return self._thread is not None and self._thread.is_alive()

    def _listen(self):
        """
        The method that does the actual listening and adding the audio to a queue.
        """
        while self.is_listening:
            try:
                audio = self._listener.listen(**self._listener_kwargs)
                logger.debug(f"Received audio: {audio}")
                self.queue.put(audio)
            except EOFError:
                logger.info("Background listener reached end of file")
                break
            except Exception:
                logger.exception("Background listener received exception")
                break

    def get(self) -> AudioSample:
        return self.queue.get()


class BaseListener:
    def __init__(self, source: BaseAudioSource):
        self.source = source

    def _determine_frame_state(self, latest_frame: AudioSample, all_frames: List[AnnotatedFrame]) -> FrameStateEnum:
        """
        Determine the state of the listener from the ``latest_frame`` that's recorded and all the frames that have been
        recorded (except for the ``latest_frame``.
        """
        raise NotImplementedError

    def _pre_process_individual_audio_sample(self, audio: AudioSample) -> AudioSample:
        """
        Apply any kind of filtering or modification of an audio sample before determining the frame state.  By default,
        it just returns the audio sample unchanged.
        """
        return audio

    def listen_frames(self, chunk_size: int = 2**14) -> List[AnnotatedFrame]:
        """
        Record and collect audio frames until the ``STOP`` frame state has been reached.  Return a list of all frames
        recorded (including the ``STOP`` frame).

        If the end of the source is reached, then listening will end regardless of what ``_determine_frame_state`` will
        stop.
        """
        all_frames = []

        frame = self.source.read(chunk_size)
        frame = self._pre_process_individual_audio_sample(frame)
        frame_state = self._determine_frame_state(frame, all_frames)

        while frame_state != FrameStateEnum.STOP:
            all_frames.append(AnnotatedFrame(frame, frame_state))
            try:
                frame = self.source.read(chunk_size)
            except EOFError:
                frame = None
                frame_state = FrameStateEnum.STOP
                break
            frame = self._pre_process_individual_audio_sample(frame)
            frame_state = self._determine_frame_state(frame, all_frames)

        if frame is not None:
            all_frames.append(AnnotatedFrame(frame, frame_state))
        return all_frames

    def _filter_audio_samples(self, all_frames: List[AnnotatedFrame]) -> List[AnnotatedFrame]:
        """
        Modify/remove audio samples before they are merged into a single audio sample.

        This method keeps all frames where the frame state was ``LISTEN`` or where the previous frame's state was
        ``LISTEN``.
        """
        previous_state = None
        saved_frames = []
        for frame in all_frames:
            if frame.state == FrameStateEnum.LISTEN or previous_state == FrameStateEnum.LISTEN:
                saved_frames.append(frame)
            previous_state = frame.state
        return saved_frames

    def _join_audio_samples(self, all_frames: List[AnnotatedFrame]) -> AudioSample:
        """
        Combine audio frames into a single audio sample.
        """
        return AudioSample.from_iterable(frame.frame for frame in all_frames)

    def _post_process_final_audio_sample(self, audio: AudioSample) -> AudioSample:
        """
        Apply any kind of filtering or modification of the final audio sample.  By default, it just returns the audio
        sample unchanged.
        """
        return audio

    def listen(self, chunk_size: int = 2**14) -> AudioSample:
        """
        This method drives the listening action by calling the other methods and returning the final result.
        """
        all_frames = self.listen_frames(chunk_size)
        all_frames = self._filter_audio_samples(all_frames)
        audio_sample = self._join_audio_samples(all_frames)
        audio_sample = self._post_process_final_audio_sample(audio_sample)

        return audio_sample

    def get(self) -> AudioSample:
        return self.listen()

    def background_listen(self, chunk_size: int = 2**14) -> BackgroundListener:
        """
        Produce a background listener object that uses this listener.
        """
        return BackgroundListener(self, listener_kwargs={"chunk_size": chunk_size})
