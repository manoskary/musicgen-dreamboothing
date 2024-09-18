from peft import PeftConfig, PeftModel
from sympy.physics.mechanics.tests.test_system import states
from transformers import AutoModelForTextToWaveform, AutoProcessor
import torch
import soundfile as sf
import os
import librosa
import argparse
import numpy as np
import pandas as pd
import maad


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--model_path", type=str, default="ylacombe/musicgen-melody-punk-lora")
    parser.add_argument("--output_dir", type=str, default="artifacts")
    parser.add_argument("--guidance_scale", type=int, default=3)
    parser.add_argument("--file_name", type=str, default="musicgen_out_0.wav")
    parser.add_argument("--current_state", type=str, default="sad")
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argumnet("--target_state", type=str, default="happy")


    args = parser.parse_args()
    return args


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


def compute_next_prompt(previous_prompt, state):
    prompt = ""

    return prompt




def main():
    args = parse_args()
    base_dir = os.path.join(os.path.dirname(__file__), args.output_dir)
    os.makedirs(base_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.device_count() > 0 and args.device >= 0 else "cpu")

    model_path = args.model_path

    config = PeftConfig.from_pretrained(model_path)
    model = AutoModelForTextToWaveform.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, model_path).to(device)

    processor = AutoProcessor.from_pretrained(model_path)
    sampling_rate = 32000
    inputs = processor(
        sampling_rate=sampling_rate,
        text=args.input_prompt,
        padding=True,
        return_tensors="pt"
    ).to(device)

    inputs["input_values"] = inputs["input_values"].half()

    audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=1503)
    generation_length = 30 # seconds
    max_new_tokens = int(1503 * (1 - args.overlap))
    sample = audio_values / np.max(np.abs(audio_values))
    index_stop = int(1 / args.overlap)

    states = globals_and_setup(args.current_state, args.target_state)

    # TODO: ~7 seconds of the source or previoussly generated audio is used as prompt, we can use this 7sec overlap to
    #  crossfade the generated audio to avoid clicks, phase shifts and volume changes after normalization.
    audios = [sample]
    text_prompt = args.input_prompt

    for i in range(10):
        sample_next = sample[- len(sample) // index_stop : ]
        text_prompt = compute_next_prompt(text_prompt)
        inputs = processor(
            audio=sample_next,
            sampling_rate=sampling_rate,
            text=,
            padding=True,
            return_tensors="pt"
        ).to(device)

        inputs["input_values"] = inputs["input_values"].half()

        audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=max_new_tokens)
        audio_values = processor.batch_decode(audio_values, padding_mask=inputs.padding_mask)
        sample = audio_values[0]
        # normalize sample
        sample = sample / np.max(np.abs(sample))

        audios.append(sample)

    final_audio = maad.util.crossfade_list(audios, fs=sampling_rate, fade_len=generation_length*args.overlap)
    sf.write(os.path.join(base_dir, f"music_medicine_{args.current_state}-{args.target_state}.wav"), sampling_rate)
    # # add the new audio to the original samples
    # audio_file_01 = np.hstack((sample[: -len(sample) // 4], audio_values[0].squeeze()))
    # audio_file_02 = np.hstack((sample[: -len(sample) // 4], audio_values[1].squeeze()))
    #
    # sampling_rate = model.config.audio_encoder.sampling_rate
    #
    # sf.write(os.path.join(base_dir, os.path.splitext(fn)[0] + "_0_continue.wav"), audio_file_01, sampling_rate)
    # sf.write(os.path.join(base_dir, os.path.splitext(fn)[0] + "_1_continue.wav"), audio_file_02, sampling_rate)
