import torch
import torchaudio
import librosa
from torcheval.metrics import FrechetAudioDistance

def preprocess_audio(input_audio):
    print(f"Input audio shape: {input_audio.shape if hasattr(input_audio, 'shape') else 'scalar'}")
    print(f"Input audio type: {type(input_audio)}")
    
    # Handle scalar input
    if isinstance(input_audio, (float, int)) or (isinstance(input_audio, torch.Tensor) and input_audio.dim() == 0):
        print("Warning: Input is a scalar. Creating a dummy waveform.") # WE ALWAYS HAVE A SCALAR INPUT VALUE HERE FOR SOME REASON
        input_audio = torch.zeros(16000)  
    
    # Ensure input is a tensor
    if not isinstance(input_audio, torch.Tensor):
        input_audio = torch.tensor(input_audio)
    
    # Ensure 2D: (channels, samples)
    if input_audio.dim() == 1:
        input_audio = input_audio.unsqueeze(0)
    
    # Set to Mono
    if input_audio.dim() == 2 and input_audio.size(0) > 1:
        input_audio = input_audio.mean(dim=0, keepdim=True)
    
    print(f"Preprocessed audio shape: {input_audio.shape}")
    return input_audio

# Load audio files
real_audio, real_audio_sr = librosa.load("test_songs/LikeLoatheIt.mp3", sr=16000)
final_audio, final_audio_sr = librosa.load("test_songs/Fribgane_Amazigh.mp3", sr=16000)

# Convert to tensors
real_audio_tensor = torch.from_numpy(real_audio).float()
final_audio_tensor = torch.from_numpy(final_audio).float()

print(f"Real audio tensor shape: {real_audio_tensor.shape}")
print(f"Final audio tensor shape: {final_audio_tensor.shape}")

# Load VGGish model
fad_base_model = torch.hub.load('harritaylor/torchvggish', 'vggish')
fad_base_model.eval()

# Initialize FAD metric
embedding_dim = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# WE GET AN ATTRIBUTE ERROR IN THE VGGISH MODEL HERE
fad_metric = FrechetAudioDistance(
    preproc=preprocess_audio,
    model=fad_base_model,
    embedding_dim=embedding_dim,
    device=device
)

# Update FAD metric
fad_metric.update(final_audio_tensor, real_audio_tensor)

# Compute FAD score
fad_score = fad_metric.compute()
print(f"FAD score: {fad_score.item()}")