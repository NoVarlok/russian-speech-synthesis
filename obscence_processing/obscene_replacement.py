import numpy as np
from scipy.io.wavfile import read


class ObsceneReplacement:
    def __init__(self, audio_path: str = 'data/dolphin_sound.wav'):
        self.max_wav_value = 32768.0
        self.hop_length = 256
        self.sample_rate, self.audio = read(audio_path)
        self.audio = np.float32(self.audio)
        self.audio = self.audio[:, 0]
