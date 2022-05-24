import os
import sys
import numpy as np
import pandas as pd
import torch
import librosa
from scipy.io.wavfile import read, write
from pysndfx import AudioEffectsChain

from tps import cleaners, Handler, load_dict, save_dict
from tps.content import ops
from tps.modules import RuEmphasizer
from tps.modules.ssml import parse_ssml_text

from obscence_processing import ObsceneSplitter, ObsceneReplacement
from ssml_wrappers import SSMLException, ssml_factory

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)
import wrappers as bw

sys.path.pop(0)


class Synthesizer:
    def __init__(self, text_handler, engine, vocoder, sample_rate,
                 obscene_words_path='data/obscene.txt',
                 obscene_replacement_path='data/dolphin_sound.wav', ):
        self.text_handler = text_handler
        try:
            stress_dict = ops.find("stress.dict", raise_exception=True)
        except FileNotFoundError:
            stress_dict = ops.download("stress.dict")
        self.emphasizer = RuEmphasizer([stress_dict, "plane"], True)

        self.engine = engine
        self.vocoder = vocoder

        self.sample_rate = sample_rate
        self.trim_top_db = 45
        self.ft_window = 1024
        self.hop_length = 256
        self.max_wav_value = 32768.0

        self.device = self.engine.device
        if self.engine.device != self.vocoder.device:
            raise RuntimeError("Both model must be placed on the same device")

        self.obscene_splitter = ObsceneSplitter(f_words_path=obscene_words_path,
                                                audio_obscene_replacement_path=obscene_replacement_path)

    def preprocess_text(self, raw_text):
        text = raw_text.lower()
        text = self.emphasizer.process_text(text)
        text = self.text_handler.process_text(
            text, cleaners.light_punctuation_cleaners, None, False,
            mask_stress=False, mask_phonemes=True
        )
        text = self.text_handler.check_eos(text)
        text_vector = self.text_handler.text2vec(text)
        text_tensor = torch.IntTensor(text_vector)
        return text_tensor

    def get_audio(self, audio_path, trim_silence=False, add_silence=False):
        sample_rate, audio = read(audio_path)
        audio = np.float32(audio / self.max_wav_value)  # faster than loading using librosa

        if sample_rate != self.sample_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(sample_rate, self.sample_rate))

        audio_ = audio.copy()

        if trim_silence:
            idxs = librosa.effects.split(
                audio_,
                top_db=self.trim_top_db,
                frame_length=self.ft_window,
                hop_length=self.hop_length,
            )

            audio_ = np.concatenate([audio_[start:end] for start, end in idxs])

        if add_silence:
            audio_ = np.append(audio_, np.zeros(5 * self.hop_length))

        audio_ = torch.FloatTensor(audio_.astype(np.float32))
        audio_ = audio_.unsqueeze(0)
        audio_ = torch.autograd.Variable(audio_, requires_grad=False)

        return audio_

    def load_mel(self, audio_path):
        audio = self.get_audio(audio_path=audio_path, trim_silence=False, add_silence=True)
        mel = self.engine.stft.mel_spectrogram(audio)
        mel = torch.unsqueeze(mel, 0).to(self.device)
        return mel

    def synthesize(self, text, gst_audio_path):
        text_tensor = self.preprocess_text(text)
        text_tensor = torch.unsqueeze(text_tensor, 0)
        text_tensor = text_tensor.to(self.device)
        gst_mel = self.load_mel(gst_audio_path).to(self.device)
        mel_outputs_postnet = self.engine(text_tensor, reference_mel=gst_mel)

        audio = self.vocoder(mel_outputs_postnet)
        audio = audio * self.max_wav_value
        audio = audio.squeeze()
        audio = audio.cpu().numpy()
        return audio

    def synthesize_filtered(self, text, gst_audio_path):
        split_text = self.obscene_splitter.split(text)
        gst_mel = self.load_mel(gst_audio_path).to(self.device)
        audio_segments = []
        for sample in split_text:
            if isinstance(sample, str):
                sample_tensor = self.preprocess_text(sample)
                sample_tensor = torch.unsqueeze(sample_tensor, 0)
                sample_tensor = sample_tensor.to(self.device)
                mel_outputs_postnet = self.engine(sample_tensor, reference_mel=gst_mel)

                audio = self.vocoder(mel_outputs_postnet)
                audio = audio * self.max_wav_value
                audio = audio.squeeze()
                audio = audio.cpu().numpy()

                audio_segments.append(audio)
            elif isinstance(sample, ObsceneReplacement):
                audio_segments.append(sample.audio)
        filtered_audio = np.concatenate(audio_segments, axis=0)
        return filtered_audio

    def synthesize_ssml(self, ssml_string, gst_audio_path):
        try:
            ssml_elements = parse_ssml_text(ssml_string)
        except SSMLException as e:
            raise e
        except Exception as e:
            raise RuntimeError("Failed to parse ssml")
        audio_segments = []
        for element in ssml_elements:
            ssml_element = ssml_factory(element)
            content, is_text = ssml_element.get_content()
            if is_text:
                content = self.synthesize(content, gst_audio_path)
            audio = ssml_element.postprocess_content(content)
            audio_segments.append(audio)
        merged_audio = np.concatenate(audio_segments, axis=0)
        return merged_audio

    def save_audio(self, audio, path, sample_rate=None):
        if sample_rate is None:
            sample_rate = self.sample_rate
        audio = audio.astype('int16')
        write(path, sample_rate, audio)

    def change_speed(self, audio, factor):
        fx = (AudioEffectsChain().speed(factor))
        changed_audio = fx(audio, sample_in=self.sample_rate)
        return changed_audio

    def change_pitch(self, audio, shift):
        fx = (AudioEffectsChain().pitch(shift))
        changed_audio = fx(audio, sample_in=self.sample_rate)
        return changed_audio


if __name__ == '__main__':
    tacotron_checkpoint_path = '/home/lyakhtin/repos/tts/final_models/tacotron-gst-apr-17-final-low-lr-natasha-train'
    vocoder_checkpoint_path = '/home/lyakhtin/repos/tts/final_models/waveglow_converted.pt'
    device = 'cuda'
    text_handler = Handler(charset='ru')
    engine = bw.Tacotron2Wrapper(model_path=tacotron_checkpoint_path, device=device)
    vocoder = bw.WaveglowWrapper(model_path=vocoder_checkpoint_path,
                                 device=device,
                                 sigma=1.0)
    # vocoder = bw.HiFiGanWrapper(model_path=vocoder_checkpoint_path, device=device)
    synthesizer = Synthesizer(text_handler=text_handler,
                              engine=engine,
                              vocoder=vocoder,
                              sample_rate=22050)
    input_data_path = '/home/lyakhtin/repos/tts/datasets/natasha_dataset/tacotron_inference_short.csv'
    src_audio_dir = '/home/lyakhtin/repos/tts/gst_wavs/'
    audio_list = ['boss-in-this-gym_fixed', 'deep-dark-fantasies_fixed', 'fuckyou', 'iam-cumming_fixed', 'laugh',
                  'what-the-hell-are-you-two-doing']
    tacotron_name = tacotron_checkpoint_path.split('/')[-1]
    vocoder_name = vocoder_checkpoint_path.split('/')[-1]
    save_directory_path = '/home/lyakhtin/repos/tts/final_models/results/' + f'{tacotron_name}_{vocoder_name}'
    if not os.path.exists(save_directory_path):
        os.mkdir(save_directory_path)
    df = pd.read_csv(input_data_path, sep='|')
    for audio_name in audio_list:
        audio_path = src_audio_dir + audio_name + '.wav'
        audio_dir = os.path.join(save_directory_path, audio_name)
        if not os.path.exists(audio_dir):
            os.mkdir(audio_dir)
        for index, row in df.iterrows():
            wav_name = row[0]
            text = row[1]
            generated_path = os.path.join(audio_dir, wav_name)
            audio = synthesizer.synthesize(text, audio_path)
            synthesizer.save_audio(audio, generated_path, 22050)
