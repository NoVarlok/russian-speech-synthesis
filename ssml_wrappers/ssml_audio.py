import re
import os
import urllib.request
from scipy.io.wavfile import read
import numpy as np
from datetime import datetime

from ssml_wrappers import SSMLElement, SSMLException


class SSMLAudio(SSMLElement):
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    def __init__(self, audio_path: str, pitch=1.0, rate=1.0, volume=1.0):
        super().__init__(pitch=pitch, rate=rate, volume=volume)
        is_url = audio_path.startswith('http://') or audio_path.startswith('https://')
        if is_url:
            if not SSMLAudio.check_url_correctness(audio_path):
                raise SSMLException(f'Invalid url: {audio_path}')
            self.filename = os.path.join('tmp',
                                         str(datetime.now()).replace(' ', '_').replace('.', '_').replace(':', '_'))
            try:
                urllib.request.urlretrieve(audio_path, self.filename)
                self.audio = self.load_audio()
            except urllib.error.HTTPError as e:
                raise SSMLException(f'Cannot get audio: {e}')
            except Exception as e:
                raise SSMLException(f'Cannot load {self.filename}')

        else:
            self.filename = audio_path

        self.audio = self.load_audio()

        if is_url and os.path.exists(self.filename):
            os.remove(self.filename)

    def get_content(self):
        return self.audio, False

    @staticmethod
    def check_url_correctness(audio_path):
        return re.match(SSMLAudio.regex, audio_path) is not None

    def load_audio(self):
        sample_rate, audio = read(self.filename)
        audio = np.float32(audio)
        audio = audio[:, 0]
        return audio
