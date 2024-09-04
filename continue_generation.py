from peft import PeftConfig, PeftModel
from transformers import AutoModelForTextToWaveform, AutoProcessor
import torch
import soundfile as sf
import os
import librosa
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--device", type=int, default=0)
parser.add_argument("--model_path", type=str, default="ylacombe/musicgen-melody-punk-lora")
parser.add_argument("--output_dir", type=str, default="artifacts")
parser.add_argument("--guidance_scale", type=int, default=3)
parser.add_argument("--file_name", type=str, default="musicgen_out_0.wav")

args = parser.parse_args()

base_dir = os.path.join(os.path.dirname(__file__), args.output_dir)
os.makedirs(base_dir, exist_ok=True)

device = torch.device(f"cuda:{args.device}" if torch.cuda.device_count() > 0 and args.device >= 0 else "cpu")

model_path = args.model_path

fn = args.file_name

if not os.path.exists(os.path.join(base_dir, fn)):
    raise FileNotFoundError(f"File {fn} not found in {base_dir}")

sample, sr = librosa.load(os.path.join(base_dir, fn), sr=32000)

sample_1 = sample[len(sample) // 4 : ]
sample_2 = sample[len(sample) // 4 : ]

config = PeftConfig.from_pretrained(model_path)
model = AutoModelForTextToWaveform.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, model_path).to(device)

processor = AutoProcessor.from_pretrained(model_path)


inputs = processor(
    audio=[sample_1, sample_2],
    sampling_rate=sr,
    text=["sad, ambient, soundscape, piano, strings, flute", "happy, upbeat, piano, strings, flute, soundscape"],
    padding=True,
    return_tensors="pt"
).to(device)

inputs["input_values"] = inputs["input_values"].half()

audio_values = model.generate(**inputs, do_sample=True, guidance_scale=1, max_new_tokens=750)
audio_values = processor.batch_decode(audio_values, padding_mask=inputs.padding_mask)

# add the new audio to the original samples
audio_file_01 = audio_file_01 = np.hstack((sample[:len(sample) // 4], audio_values[0].squeeze()))
audio_file_02 = np.hstack((sample[:len(sample) // 2], audio_values[1].squeeze()))

sampling_rate = model.config.audio_encoder.sampling_rate
audio_values = audio_values.cpu().float().numpy()

sf.write(os.path.join(base_dir, os.path.splitext(fn)[0] + "_0_continue.wav"), audio_file_01.T, sampling_rate)
sf.write(os.path.join(base_dir, os.path.splitext(fn)[0] + "_1_continue.wav"), audio_file_02.T, sampling_rate)
