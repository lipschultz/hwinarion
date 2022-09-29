from typing import List, Callable, Optional

from hwinarion.audio.base import BaseAudioSource, AudioSample, TimeType
from hwinarion.listeners.base import BaseListener, AnnotatedFrame, FrameStateEnum


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
