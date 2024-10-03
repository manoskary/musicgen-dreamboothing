import numpy as np
import laion_clap
from scipy.linalg import sqrtm
import librosa
import numpy
import torch

class FADMetric:
    def __init__(self):
        # Load CLAP model
        self.model = laion_clap.CLAP_Module(enable_fusion=False)
        self.model.load_ckpt() # download the default pretrained checkpoint.
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def extract_features(self, audio, sr=48000, window_size=1.0, hop_size=0.5):
        # Convert window and hop sizes from seconds to samples
        window_samples = int(window_size * sr)
        hop_samples = int(hop_size * sr)

        # Pad audio if it's shorter than the window size
        if len(audio) < window_samples:
            audio = np.pad(audio, (0, window_samples - len(audio)))

        # Extract features using sliding window
        features = []
        for start in range(0, len(audio) - window_samples + 1, hop_samples):
            window = audio[start:start + window_samples]
            window = (window - window.mean()) / window.std()
            window_tensor = torch.from_numpy(window).float().unsqueeze(0)
            if torch.cuda.is_available():
                window_tensor = window_tensor.cuda()
            with torch.no_grad():
                feature = self.model.get_audio_embedding_from_data(window_tensor, sr)
            features.append(feature.squeeze().cpu().numpy())
        features = (features - np.mean(features)) / np.std(features)

        return np.array(features)



    def calculate_fad(self, real_audio, generated_audio):
        real_waveform, real_sr = librosa.load(real_audio, sr=48000, mono=True) #16kHz for vggish, #48kHz for CLAP
        generated_waveform, generated_sr = librosa.load(generated_audio, sr=48000, mono=True)

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

        real_features = (real_features-numpy.min(real_features))/(numpy.max(real_features)-numpy.min(real_features))
        gen_features = (gen_features-numpy.min(gen_features))/(numpy.max(gen_features)-numpy.min(gen_features))


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
        