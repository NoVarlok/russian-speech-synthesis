import abc
from pysndfx import AudioEffectsChain
import numpy as np


class SSMLElement(abc.ABC):
    max_wav_value = 32768.0
    sample_rate = 22050

    def __init__(self, pitch=1.0, rate=1.0, volume=1.0):
        self.pitch = pitch
        self.rate = rate
        self.volume = volume

    @abc.abstractmethod
    def get_content(self):
        pass

    def postprocess_content(self, audio):
        processed_audio = audio / SSMLElement.max_wav_value
        processed_audio = self.change_pitch(processed_audio)
        processed_audio = self.change_volume(processed_audio)
        processed_audio = self.change_rate(processed_audio)
        processed_audio = processed_audio * SSMLElement.max_wav_value
        return processed_audio

    def change_pitch(self, audio):
        fx = (AudioEffectsChain().pitch(self.pitch))
        changed_audio = fx(audio, sample_in=self.sample_rate)
        return changed_audio

    def change_volume(self, audio):
        changed_audio = audio * self.volume
        changed_audio = np.clip(changed_audio, -1.0, 1.0)
        return changed_audio

    def change_rate(self, audio):
        fx = (AudioEffectsChain().speed(self.rate))
        changed_audio = fx(audio, sample_in=self.sample_rate)
        return changed_audio
