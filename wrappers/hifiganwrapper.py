import sys
import torch
import numpy as np

_hifigan_path = sys.path[0]

# from models import Generator
import env
import models



class HiFiGanWrapper:
    def __init__(self, model_path, device):
        self.device = torch.device("cpu" if not torch.cuda.is_available() else device)
        self.dtype = torch.float

        model = torch.load(model_path)
        self.model = torch.load(model_path, map_location=self.device)
        self.model.device = self.device

        self.model.eval().to(device=self.device, dtype=self.dtype)
        self.model.remove_weight_norm()

    def __call__(self, spectrogram):
        with torch.no_grad():
            audio = self.model(spectrogram)

        return audio
