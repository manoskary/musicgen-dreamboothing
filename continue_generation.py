import random
from peft import PeftConfig, PeftModel
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
    """
    Set up for the argument parser.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--model_path", type=str, default="ylacombe/musicgen-melody-punk-lora")
    parser.add_argument("--output_dir", type=str, default="artifacts")
    parser.add_argument("--guidance_scale", type=int, default=3)
    parser.add_argument("--file_name", type=str, default="musicgen_out_0.wav")
    parser.add_argument("--current_state", type=str, default="sad")
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--length", type=int, default=15, help="generation length in minutes")
    parser.add_argument("--target_state", type=str, default="happy")
    parser.add_argument("--input_prompt", type=str, default="sad piano with ambient sounds")

    args = parser.parse_args()
    return args


def globals_and_setup(init_state, final_state):
    """
    This functions computes the states that need to be traversed from the init state to the final state.

    """
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

    # multiply the number of states by the steps per state


    return steps_per_state


def get_prob_list_of_states(states, num_steps_per_state):
    """
    Given the list of states to pass through and the estimate of how many generation steps should every state last,
    create a full list of the states with repeats and potential regressions.

    """
    result = []
    for i, num_steps in enumerate(num_steps_per_state):
        if i == 0:
            result = [states[0]]*num_steps
        else:
            for j in range(num_steps):
                # possibility to regress to previous state by rolling the dice 1 out of 6
                # TODO: potentially something more sofisticated for the zig zag regression motion
                if random.randint(0, 6) == 1 and j > 0:
                    result.append(states[i-1])
                else:
                    result.append(states[i])
    return result

def main():
    args = parse_args()
    # default output should be in folder ./artifacts/
    base_dir = os.path.join(os.path.dirname(__file__), args.output_dir)
    os.makedirs(base_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.device_count() > 0 and args.device >= 0 else "cpu")

    model_path = args.model_path

    # to recreate it or modify it check out the script under data/dataset_stats.py
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'jamendo_stats.csv'), header=[0, 1])

    # this function needs to be here to use the df as global.
    def compute_next_prompt(previous_prompt, next_state):
        """
        given a prompt, identify instruments, mood and genre

        compute the next state (mood) and then accordingly modify instruments and genre;
        """
        # TODO: maybe it will be better to use mood instead of emotion for the prompt, given that the dataset had mood tags.

        words = previous_prompt.replace(",", "").split(" ")

        cur_instruments = []
        cur_genre = []
        for w in words:
            # find current instruments
            if w in df["instrument"].columns:
                cur_instruments.append(w)

            # find current genre
            if w in df["genre"].columns:
                cur_genre.append(w)

        # roll dice whether to change the instrumentation (1 to 6)
        # TODO: find a better way to change the instrumentation
        if random.randint(0, 6) == 1:
            # Get from the dataset statistics all the instrumentations that had the same emotion as the next state.
            instrument_df = df["instrument"][df["emotion"][next_state] == True]
            # randomly pick one
            rand_index = random.randint(0, len(instrument_df))
            # fetch that random row
            row = instrument_df.iloc[rand_index]
            # find which instruments are present
            cur_instruments = row[row != 0].keys().to_list()


        # roll dice whether to change the genre (1 to 6):
        # TODO: find a better way to change the genre
        if random.randint(0, 6) == 1:
            # Get from the dataset statistics the genre that had the same emotion as the next state.
            genre_df = df["genre"][df["emotion"][next_state] == True]
            # randomly pick one
            rand_index = random.randint(0, len(genre_df))
            # fetch that random row
            row = genre_df.iloc[rand_index]
            # find which genre is present
            cur_genre = row[row != 0].keys().to_list()

        # next prompt is a combination of the next state, instrumentation (random or previous) and genre separated by comma
        # e.g. sad, piano, guitar, 80s
        prompt = [next_state] + cur_instruments + cur_genre
        prompt = ", ".join(prompt)
        return prompt

    # set up model staff
    config = PeftConfig.from_pretrained(model_path)
    model = AutoModelForTextToWaveform.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, model_path).to(device)
    processor = AutoProcessor.from_pretrained(model_path)
    sampling_rate = 32000
    inputs = processor(
        sampling_rate=sampling_rate,
        text=[args.input_prompt],
        padding=True,
        return_tensors="pt"
    ).to(device)


    audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=1503).squeeze()
    generation_length = 30 # seconds
    # 1503 tokes is 30 seconds of generation for MusicGen
    max_new_tokens = int(1503 * (1 - args.overlap))
    # Normalize the audio to keep a consistent volume level and avoid fade outs.
    sample = audio_values / np.max(np.abs(audio_values))
    # index_stop is used to feed the appropriate proportion of every previously generated as condition for the next
    index_stop = int(1 / args.overlap)

    # Compute emotion labels for the entire generation process.
    states = globals_and_setup(args.current_state, args.target_state)
    num_steps_per_state = compute_num_steps(args.length, len(states))
    states_with_repeat = get_prob_list_of_states(states, num_steps_per_state)


    audios = [sample]
    text_prompt = args.input_prompt

    for i in range(len(states_with_repeat)):
        sample_next = sample[- len(sample) // index_stop : ]
        text_prompt = compute_next_prompt(text_prompt, states_with_repeat[i])
        inputs = processor(
            audio=sample_next,
            sampling_rate=sampling_rate,
            text=[text_prompt],
            padding=True,
            return_tensors="pt"
        ).to(device)

        inputs["input_values"] = inputs["input_values"].half()

        audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=max_new_tokens).squeeze()
        audio_values = processor.batch_decode(audio_values, padding_mask=inputs.padding_mask)
        sample = audio_values[0]
        # normalize sample
        sample = sample / np.max(np.abs(sample))
        audios.append(sample)

    # This utility applies cross fade for all audio segments.
    final_audio = maad.util.crossfade_list(audios, fs=sampling_rate, fade_len=generation_length*args.overlap)
    sf.write(os.path.join(base_dir, f"music_medicine_{args.current_state}-{args.target_state}.wav"), final_audio, sampling_rate)
    # # add the new audio to the original samples
    # audio_file_01 = np.hstack((sample[: -len(sample) // 4], audio_values[0].squeeze()))
    # audio_file_02 = np.hstack((sample[: -len(sample) // 4], audio_values[1].squeeze()))
    #
    # sampling_rate = model.config.audio_encoder.sampling_rate
    #
    # sf.write(os.path.join(base_dir, os.path.splitext(fn)[0] + "_0_continue.wav"), audio_file_01, sampling_rate)
    # sf.write(os.path.join(base_dir, os.path.splitext(fn)[0] + "_1_continue.wav"), audio_file_02, sampling_rate)


if __name__ == "__main__":
    main()