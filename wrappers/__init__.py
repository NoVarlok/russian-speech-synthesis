import os
import sys

backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../submodules")

import_path = os.path.join(backend_path, "tacotron2")
sys.path.insert(0, import_path)
from .tacotron2wrapper import Tacotron2Wrapper

sys.path.pop(0)

import_path = os.path.join(backend_path, "waveglow")
sys.path.insert(0, import_path)
from .waveglowwrapper import WaveglowWrapper

sys.path.pop(0)

import_path = os.path.join(backend_path, "hifigan")
sys.path.insert(0, import_path)
from .hifiganwrapper import HiFiGanWrapper

sys.path.pop(0)
