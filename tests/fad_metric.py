import numpy as np
import laion_clap
from scipy.linalg import sqrtm
import librosa
import torch
import torchaudio
import os

class FADMetric:
    def __init__(self):
        #self.model = laion_clap.CLAP_Module(enable_fusion=False)
        #self.model.load_ckpt() # download the default pretrained checkpoint.
        # Load VGGish model
        self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.model.eval()

    def load_audio(self, file_path, sr=48000):
        """
        Load an audio file using librosa.
        
        :param file_path: Path to the audio file
        :param sr: Target sampling rate. Default is 48000 as required by LAION-CLAP
        :return: Numpy array of audio samples, Sampling rate
        """
        audio, sr = librosa.load(file_path, sr=sr)
        return audio, sr

    def extract_features(self, audio):
        """
        Extract features using the LAION-CLAP model.
        
        :param audio: Either a file path or a numpy array of audio samples
        :return: Extracted features
        """
        if isinstance(audio, str) and os.path.isfile(audio):
            # If audio is a file path, use get_audio_embedding_from_filelist
            return self.model.get_audio_embedding_from_filelist([audio])[0]
        elif isinstance(audio, np.ndarray):
            # If audio is already loaded as a numpy array, use get_audio_embedding
            return self.model.get_audio_embedding(audio)
        else:
            raise ValueError("Invalid audio input. Must be either a file path or a numpy array.")

    def calculate_fad(self, real_audio, generated_audio):
        real_features = self.extract_features(real_audio)
        gen_features = self.extract_features(generated_audio)

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