from fad_metric import FADMetric

def main():
    fad_calculator = FADMetric()
    
    # Load your real and generated audio data
    real_audio_path = "test_songs/Fribgane_Amazigh.mp3"
    generated_audio_path = "test_songs/LikeLoatheIt.mp3"
    
    # You can now pass the file paths directly to calculate_fad
    fad_score = fad_calculator.calculate_fad(real_audio_path, generated_audio_path)
    print(f"FAD score: {fad_score}")

if __name__ == "__main__":
    main()