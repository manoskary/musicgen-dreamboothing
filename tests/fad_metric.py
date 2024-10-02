import numpy as np
import laion_clap
from scipy.linalg import sqrtm
import librosa
import numpy
import torch
import torchaudio
import os

class FADMetric:
    def __init__(self):
        # Load VGGish model
        self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.model.eval()

    def extract_features(self, audio):
        return self.model.forward(audio, fs=16000)

    def calculate_fad(self, real_audio, generated_audio):
        real_waveform, real_sr = librosa.load(real_audio, sr=16000, mono=True)
        generated_waveform, generated_sr = librosa.load(generated_audio, sr=16000, mono=True)
        generated_waveform=numpy.random.normal(2*real_waveform+2,20)

        len1, len2 = len(generated_waveform), len(real_waveform)
        min_len = min(len1, len2)
        real_waveform = real_waveform[:min_len]
        generated_waveform = generated_waveform[:min_len]

        real_features = self.extract_features(real_waveform)
        gen_features = self.extract_features(generated_waveform)

        if isinstance(real_features, torch.Tensor):
            real_features = real_features.detach().cpu().numpy()
        if isinstance(gen_features, torch.Tensor):
            gen_features = gen_features.detach().cpu().numpy()


        # Calculate mean and covariance
        mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu_gen, sigma_gen = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
        # Calculate FAD
        diff = mu_real - mu_gen
        covmean = sqrtm(sigma_real.dot(sigma_gen))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fad = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2*covmean)

        return fad
        