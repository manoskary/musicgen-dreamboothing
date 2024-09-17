from peft import PeftConfig, PeftModel
from transformers import AutoModelForTextToWaveform, AutoProcessor
import torch
import soundfile as sf
import os
import librosa
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("--device", type=int, default=0)
parser.add_argument("--model_path", type=str, default="ylacombe/musicgen-melody-punk-lora")
parser.add_argument("--output_dir", type=str, default="artifacts")
parser.add_argument("--guidance_scale", type=int, default=3)
parser.add_argument("--file_name", type=str, default="musicgen_out_0.wav")
parser.add_argument("--current_state", type=str, default="sad")
parser.add_argumnet("--target_state", type=str, default="happy")

args = parser.parse_args()


def globals_and_setup(init_state, final_state):
    emotions = np.array([
        "bored", "depressed", "sad", "upset", "stressed",
        "nervous", "tense", "alert", "excited", "elated",
        "happy", "contented", "serene", "relaxed", "calm"])
    assert init_state in emotions
    assert final_state in emotions
    init_index = emotions.index(init_state)
    final_index = emotions.index(final_state)
    assert init_index < final_index
    return emotions[init_index, final_index]


def compute_num_steps(total_time, num_states, overlap=0.25, step_time=30):
    """
    Compute the average number of steps required for every state.

    Parameters
    ----------
    total_time: int
        The total time in minutes of the generation
    num_states: int
        The number of different emotion states
    overlap: float
        The amount of content from the generation excerpt of the previous step used for conditioning
    step_time: int
        The generation time in seconds for every generation step.

    Returns
    -------
    float
        The average number of steps required for each state.
    """
    # Convert total time from minutes to seconds
    total_time_seconds = total_time * 60

    # Calculate net new content generated per step (accounting for overlap)
    net_content_per_step = step_time * (1 - overlap)

    # Total net content needed per state
    # Total time is equally divided among states, with any leftover time added to the final state
    base_state_time = total_time_seconds // num_states
    leftover_time = total_time_seconds % num_states
    state_times = [base_state_time] * num_states
    state_times[-1] += leftover_time  # Add any leftover time to the final state

    # Calculate the number of steps for each state
    steps_per_state = np.array([state_time / net_content_per_step for state_time in state_times])

    return steps_per_state


base_dir = os.path.join(os.path.dirname(__file__), args.output_dir)
os.makedirs(base_dir, exist_ok=True)

device = torch.device(f"cuda:{args.device}" if torch.cuda.device_count() > 0 and args.device >= 0 else "cpu")

model_path = args.model_path

fn = args.file_name

if not os.path.exists(os.path.join(base_dir, fn)):
    raise FileNotFoundError(f"File {fn} not found in {base_dir}")

sample, sr = librosa.load(os.path.join(base_dir, fn), sr=32000)

# normalize the audio sample to [-1, 1]
sample = sample / np.max(np.abs(sample))


# Input prompt:
input_prompt = "sad ambient soundscape piano strings flute"

# Output prompt:
output_prompt = "happy upbeat piano strings flute soundscape"

# TODO: ~7 seconds of the source or previoussly generated audio is used as prompt, we can use this 7sec overlap to
#  crossfade the generated audio to avoid clicks, phase shifts and volume changes after normalization.
audios = [(sample[- len(sample) // 4 : ], sample[- len(sample) // 4 : ])]
for i in range(10):
    sample_1 = sample[- len(sample) // 4 : ]
    sample_2 = sample[- len(sample) // 4 : ]

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

    audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=1100)
    audio_values = processor.batch_decode(audio_values, padding_mask=inputs.padding_mask)
    sample = audio_values[0]

    audios.append((audio_values[0][-len(audio_values[0]) // 4:], audio_values[1][-len(audio_values[1]) // 4:]))

# add the new audio to the original samples
audio_file_01 = np.hstack((sample[: -len(sample) // 4], audio_values[0].squeeze()))
audio_file_02 = np.hstack((sample[: -len(sample) // 4], audio_values[1].squeeze()))

sampling_rate = model.config.audio_encoder.sampling_rate

sf.write(os.path.join(base_dir, os.path.splitext(fn)[0] + "_0_continue.wav"), audio_file_01, sampling_rate)
sf.write(os.path.join(base_dir, os.path.splitext(fn)[0] + "_1_continue.wav"), audio_file_02, sampling_rate)
