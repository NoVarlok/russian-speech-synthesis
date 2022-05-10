from ssml_wrappers import SSMLElement, SSMLException
import numpy as np


class SSMLSayAs(SSMLElement):
    @staticmethod
    def interpret_as_characters(text: str):
        text = text.upper()
        return ' '.join(list(text)), True

    @staticmethod
    def interpret_as_expletive(text: str):
        duration = SSMLSayAs.symbol_duration * len(text)
        frames = max(int(duration * SSMLElement.sample_rate), 1)
        silence = np.zeros(frames, dtype=np.float32)
        return silence, False

    symbol_duration = 0.05
    interpret_as_mapper = {
        'characters': interpret_as_characters.__func__,
        'expletive': interpret_as_expletive.__func__,
    }

    def __init__(self, interpret_as, content, pitch=1.0, rate=1.0, volume=1.0):
        super().__init__(pitch=pitch, rate=rate, volume=volume)
        self.interpreter_function = SSMLSayAs.interpret_as_mapper.get(interpret_as, None)
        if self.interpreter_function is None:
            raise SSMLException(
                f'Invalid "interpret-as" value: {interpret_as}. Supported values: ["characters", "expletive"]')
        self.interpreted_content, self.is_text = self.interpreter_function(content)

    def get_content(self):
        return self.interpreted_content, self.is_text
