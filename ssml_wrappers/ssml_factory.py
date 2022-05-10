from ssml_wrappers import SSMLAudio, SSMLBreak, SSMLException, SSMLElement, SSMLSayAs, SSMLText
from tps.modules.ssml.elements import Text, Pause, Audio, SayAs
from typing import Union


def ssml_factory(element: Union[Text, Pause, Audio, SayAs]):
    if isinstance(element, Text):
        return SSMLText(element.value, pitch=element.pitch, rate=element.rate, volume=element.volume)
    if isinstance(element, Pause):
        return SSMLBreak(element.seconds)
    if isinstance(element, Audio):
        return SSMLAudio(element.src, pitch=element.pitch, rate=element.rate, volume=element.volume)
    if isinstance(element, SayAs):
        return SSMLSayAs(interpret_as=element.interpret_as, content=element.text, pitch=element.pitch,
                         rate=element.rate, volume=element.volume)
    raise SSMLException("Invalid argument's type")



