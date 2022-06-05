# RUSSIAN-SPEECH-SYNTHESIS

RUSSIAN-SPEECH-SYNTHESIS is a end-to-end speech syntthesis solution based on [Tacotron 2](https://arxiv.org/abs/1712.05884) with [Global-Style-Token](https://arxiv.org/abs/1803.09017) as a mel-spectrongram prediction network and [Waveglow](https://arxiv.org/abs/1811.00002) or [HiFi-GAN](https://arxiv.org/abs/2010.05646) as a vocoder.

Pretrained models can be downloaded from this [link](https://drive.google.com/drive/folders/1SlafMVIwIBhFD4lgISv-yc0lhcpmbLKx?usp=sharing).

## Installation

sudo apt update
sudo apt install sox

```bash
$ sudo apt update
$ sudo apt install sox
$ pip install -r requirements.txt
```

## How to use

The speech synthesis is performed by the Synthesizer class, which implements 3 methods for speech synthesis:
* synthesize(text, gst_audio_path) - normal speech synthesis
* synthesize_filtered(text, gst_audio_path) - speech synthesis with filtering of absentee speech
* synthesize_ssml(ssml_string, gst_audio_path) - speech synthesis based on ssml markup