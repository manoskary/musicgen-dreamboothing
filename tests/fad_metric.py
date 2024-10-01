import numpy as np
import laion_clap
from scipy.linalg import sqrtm
import librosa
import torch
import torchaudio
import os

class FADMetric:
    def __init__(self):
        # Load VGGish model
        self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.model.eval()

    def extract_features(self, audio):
        return self.model.forward(audio)

    def calculate_fad(self, real_audio, generated_audio):
        real_waveform = librosa.load(real_audio, sr=16000, mono=True)
        generated_audio = librosa.load(generated_audio, sr=16000, mono=True)

        real_features = self.extract_features(real_waveform)
        gen_features = self.extract_features(generated_waveform)

        # Calculate mean and covariance
        mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu_gen, sigma_gen = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)

        # Calculate FAD
        diff = mu_real - mu_gen
        import pdb; pdb.set_trace()
        covmean = sqrtm(sigma_real.dot(sigma_gen))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fad = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2*covmean)

        return fad