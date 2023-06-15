import os
import sys

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)
import wrappers as bw
sys.path.pop(0)

from tps import cleaners, Handler, load_dict, save_dict
from synthesizer import Synthesizer

import telebot
import string
import random


if __name__ == '__main__':
    tacotron_checkpoint_path = '/home/lyakhtin/repos/tts/final_models/tacotron-gst-apr-17-unfrozen-gst'
    vocoder_checkpoint_path = '/home/lyakhtin/repos/tts/final_models/hifi-gan-v1-fine-tuned'
    device = 'cuda'
    text_handler = Handler(charset='ru')
    engine = bw.Tacotron2Wrapper(model_path=tacotron_checkpoint_path, device=device)
    # vocoder = bw.WaveglowWrapper(model_path=vocoder_checkpoint_path,
    #                              device=device,
    #                              sigma=1.0)
    vocoder = bw.HiFiGanWrapper(model_path=vocoder_checkpoint_path, device=device)
    synthesizer = Synthesizer(text_handler=text_handler,
                              engine=engine,
                              vocoder=vocoder,
                              sample_rate=22050)

    bot = telebot.TeleBot("6134775989:AAFloeXjyU34bfutmj__9uU-mfQVXLrAgac")


    def randomword(length):
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(length))

    def synthes(text):
        reference_audio_path = '/home/lyakhtin/repos/tts/datasets/natasha_dataset/wavs/000009.wav'
        save_path = '/home/lyakhtin/repos/hse/random_words/' + randomword(10) + '.wav'
        audio = synthesizer.synthesize(text, reference_audio_path)
        synthesizer.save_audio(audio, save_path, 22050)
        print('saved path:', save_path)
        return save_path

    @bot.message_handler(content_types=['text'])
    def get_text_messages(message):
        print('message:')
        print(message.text)
        bot.send_audio(message.from_user.id, open(synthes(message.text), 'rb'))
    
    print('Telegram bot is running...')
    bot.polling(none_stop=True, interval=0)
