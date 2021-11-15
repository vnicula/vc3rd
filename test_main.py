import argparse
from math import ceil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import pyaudio
import time
import torch
import torch.nn as nn
import torchaudio
from torch import Tensor
from torch.nn.functional import pad
from parallel_wavegan.utils import load_model
from data import Wav2Mel

CHANNELS = 1

attr_d = {
    "segment_size": 8000, #8192,
    "num_mels": 80,
    "num_freq": 1025,
    "n_fft": 1024,
    "hop_size": 256,
    "win_size": 1024,

    "sampling_rate": 16000,

    "fmin": 0,
    "fmax": 8000,
    "fmax_for_loss": 0,
}

def chunks(lst: List, n: int) -> List[List]:
    for i in range(0, len(lst), n):
        yield lst[i : (i + n)]


def pad_seq(x: Tensor, base: int = 32) -> Tuple[Tensor, int]:
    len_out = int(base * ceil(float(len(x)) / base))
    len_pad = len_out - len(x)
    assert len_pad >= 0
    return pad(x, (0, 0, 0, len_pad), "constant", 0), len_pad


def get_embed(encoder: nn.Module, mel: Tensor) -> Tensor:
    emb = encoder(mel[None, :])
    return emb

def load_vocoder(vocoder_path):
    # load vocoder
    vocoder = load_model(vocoder_path)
    vocoder.remove_weight_norm()
    vocoder = vocoder.eval()
    # print('vocoder', vocoder)
    return vocoder    

p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    print(p.get_device_info_by_index(i))

#prepare models
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(device)
model = torch.jit.load('model.pt').to(device)
# vocoder = torch.jit.load('vocoder.pt').to(device)
vocoder = load_vocoder("H:/work/vconv/pretrained_model/arctic_slt_parallel_wavegan.v1/checkpoint-400000steps.pkl").to(device)
wav2mel = Wav2Mel()

tgt, tgt_sr = torchaudio.load('wavs/p225_001.wav')
tgt_mel = wav2mel(tgt, tgt_sr).to(device)
tgt_emb = get_embed(model.speaker_encoder, tgt_mel)

def callback(in_data, frame_count, time_info, status):
    data = np.frombuffer(in_data, dtype=np.float32)
    data = torch.from_numpy(data).float().unsqueeze(0)
    
    src_mel = wav2mel(data, attr_d["sampling_rate"]).to(device)
    src_emb = get_embed(model.speaker_encoder, src_mel)
    print('mel shape:', src_mel.shape)
    src_mel, len_pad = pad_seq(src_mel)
    src_mel = src_mel[None, :]
    with torch.no_grad():
        _, mel, _ = model(src_mel, src_emb, tgt_emb)
    mel = mel[0, :, :] if len_pad == 0 else mel[0, :-len_pad, :]
    with torch.no_grad():
        # wav = vocoder.generate([mel])[0].data.cpu().numpy()
        # wav = vocoder.generate([src_mel])[0].data.cpu().numpy()
        wav = vocoder.inference(c=mel, normalize_before=False)
        print('Wave shape: ', wav.shape)
        wav = wav.view(-1).cpu().numpy()

    return (wav[:attr_d["segment_size"]], pyaudio.paContinue)


stream = p.open(format=pyaudio.paFloat32,
                channels=CHANNELS,
                rate=attr_d["sampling_rate"],
                # rate=24000,
                input=True,
                output=True,
                frames_per_buffer=attr_d["segment_size"],
                # frames_per_buffer=24000,
                stream_callback=callback)

print("Starting to listen.")
stream.start_stream()

while stream.is_active():
    time.sleep(0.1)

stream.stop_stream()
stream.close()

p.terminate()
