import pymorphy2
import nltk
from typing import List

from obscence_processing.obscene_replacement import ObsceneReplacement


class ObsceneSplitter:
    def __init__(self,
                 f_words_path: str = 'data/obscene.txt',
                 audio_obscene_replacement_path: str = 'data/dolphin_sound.wav'):
        with open(f_words_path, 'r') as file:
            self.f_words = [word.strip() for word in file if word]
            self.f_words = set(self.f_words)
        self.obscene_replacement = ObsceneReplacement(audio_obscene_replacement_path)
        self.morph = pymorphy2.MorphAnalyzer()

    def split(self, text: str):
        tokenized_text = nltk.word_tokenize(text)
        normalized_tokenized_text = [self.morph.parse(word.lower())[0].normal_form for word in tokenized_text]
        split_sentence = self.split_by_obscene_language(tokenized_text=tokenized_text,
                                                        normalized_tokenized_text=normalized_tokenized_text)
        return split_sentence

    def split_by_obscene_language(self, tokenized_text: List[str], normalized_tokenized_text: List[str]):
        sentences = []
        for word, normal_word in zip(tokenized_text, normalized_tokenized_text):
            if normal_word not in self.f_words:
                if not sentences or isinstance(sentences[-1], ObsceneReplacement):
                    sentences.append([])
                sentences[-1].append(word)
            else:
                sentences.append(self.obscene_replacement)
        if len(sentences) and sentences[0] == []:
            sentences = sentences[1:]
        sentences = [' '.join(sentence) if isinstance(sentence, list) else sentence for sentence in sentences]
        return sentences
