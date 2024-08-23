from peft import PeftConfig, PeftModel
from transformers import AutoModelForTextToWaveform, AutoProcessor
import torch
import soundfile as sf
import os
import librosa
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--device", type=int, default=0)
parser.add_argument("--model_path", type=str, default="ylacombe/musicgen-melody-punk-lora")
parser.add_argument("--output_dir", type=str, default="artifacts")
parser.add_argument("--guidance_scale", type=int, default=3)

args = parser.parse_args()

base_dir = os.path.join(os.path.dirname(__file__), args.output_dir)
os.makedirs(base_dir, exist_ok=True)

device = torch.device(f"cuda:{args.device}" if torch.cuda.device_count()>0 else "cpu")

model_path = args.model_path

sample, sr = librosa.load(os.path.join(base_dir, "musicgen_out_0.wav"), sr=32000)

sample_1 = sample[len(sample) // 4 : ]
sample_2 = sample[len(sample) // 2 : ]

config = PeftConfig.from_pretrained(model_path)
model = AutoModelForTextToWaveform.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, model_path).to(device)

processor = AutoProcessor.from_pretrained(model_path)


inputs = processor(
    audio=[sample_1, sample_2],
    sampling_rate=sr,
    text=["80s punk and pop track with bassy drums and synth happy", "punk bossa nova sad"],
    padding=True,
    return_tensors="pt"
).to(device)

inputs["input_values"] = inputs["input_values"].half()

audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=1024)
audio_values = processor.batch_decode(audio_values, padding_mask=inputs.padding_mask)

# add the new audio to the original samples
audio_values[0] = torch.cat([torch.tensor(sample[:len(sample) // 4]).half().to(device), audio_values[0]], dim=0)
audio_values[1] = torch.cat([torch.tensor(sample[:len(sample) // 2]).half().to(device), audio_values[1]], dim=0)

sampling_rate = model.config.audio_encoder.sampling_rate
audio_values = audio_values.cpu().float().numpy()

sf.write(os.path.join(base_dir, "musicgen_out_0_continue.wav"), audio_values[0].T, sampling_rate)
sf.write(os.path.join(base_dir, "musicgen_out_1_continue.wav"), audio_values[1].T, sampling_rate)
