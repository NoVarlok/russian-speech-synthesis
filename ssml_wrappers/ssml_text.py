from ssml_wrappers import SSMLElement, SSMLException
import numpy as np


class SSMLText(SSMLElement):
    def __init__(self, text, pitch=1.0, rate=1.0, volume=1.0):
        super().__init__(pitch=pitch, rate=rate, volume=volume)
        self.text = text

    def get_content(self):
        return self.text, True
