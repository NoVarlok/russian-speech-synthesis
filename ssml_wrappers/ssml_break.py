from ssml_wrappers import SSMLElement, SSMLException
import numpy as np


class SSMLBreak(SSMLElement):
    def __init__(self, duration):
        super().__init__()
        # self.duration = SSMLBreak.parse_duration_string(duration)
        self.duration = duration
        frame_count = int(self.duration * SSMLElement.sample_rate)
        if frame_count == 0:
            raise SSMLException(f'Invalid break duration')
        self.audio = np.zeros(frame_count, dtype=np.float32)

    def get_content(self):
        return self.audio, False

    def postprocess_content(self, audio):
        return audio

    @staticmethod
    def parse_duration_string(duration: str):
        if duration.endswith('ms'):
            try:
                duration_in_sec = float(duration[:-2]) / 1000
                return duration_in_sec
            except Exception as e:
                SSMLException(f'Invalid breaks duration: {duration}')
        if duration.endswith('s'):
            try:
                duration_in_sec = float(duration[:-1])
                return duration_in_sec
            except Exception as e:
                SSMLException(f'Invalid breaks duration: {duration}')
        raise SSMLException(f'Invalid breaks duration: {duration}')
