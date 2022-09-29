from typing import List

from hwinarion.audio.base import AudioSample, TimeType
from hwinarion.listeners.base import AnnotatedFrame, FrameStateEnum


class BaseStateLabeler:
    def __call__(self, latest_frame: AudioSample, all_frames: List[AnnotatedFrame]) -> FrameStateEnum:
        raise NotImplementedError


class TimeBasedStateLabeler(BaseStateLabeler):
    def __init__(self, total_duration: TimeType):
        super().__init__()
        self.total_duration = total_duration

    def __call__(self, latest_frame: AudioSample, all_frames: List[AnnotatedFrame]) -> FrameStateEnum:
        duration = sum(frame.frame.n_seconds for frame in all_frames) + latest_frame.n_seconds
        if duration < self.total_duration and len(latest_frame) > 0:
            return FrameStateEnum.LISTEN
        else:
            return FrameStateEnum.STOP


class SilenceBasedStateLabeler(BaseStateLabeler):
    def __init__(self, silence_threshold_rms: int = 500):
        super().__init__()
        self.silence_threshold_rms = silence_threshold_rms

    def __call__(self, latest_frame: AudioSample, all_frames: List[AnnotatedFrame]) -> FrameStateEnum:
        if latest_frame.rms > self.silence_threshold_rms:
            return FrameStateEnum.LISTEN
        else:
            if len(all_frames) == 0 or all_frames[-1].state == FrameStateEnum.PAUSE:
                # Haven't started listening yet
                return FrameStateEnum.PAUSE
            else:
                return FrameStateEnum.STOP


