import os
from fad_metric import FADMetric
from itertools import combinations
import numpy as np

def get_audio_files(directory):
    #Get all .wav files in the given directory.
    return [f for f in os.listdir(directory) if f.endswith('.wav')]

def calculate_fad_for_directory(fad_calculator, directory):
    #Calculate FAD scores for all combinations of songs in a directory.
    audio_files = get_audio_files(directory)
    results = []
    for file1, file2 in combinations(audio_files, 2):
        path1 = os.path.join(directory, file1)
        path2 = os.path.join(directory, file2)
        fad_score = fad_calculator.calculate_fad(path1, path2)
        results.append((file1, file2, fad_score))
    return results

def calculate_cross_genre_fad(fad_calculator, dir1, dir2):
    #Calculate FAD scores between songs from two different directories.
    files1 = get_audio_files(dir1)
    files2 = get_audio_files(dir2)
    results = []
    for file1 in files1:
        for file2 in files2:
            path1 = os.path.join(dir1, file1)
            path2 = os.path.join(dir2, file2)
            fad_score = fad_calculator.calculate_fad(path1, path2)
            results.append((file1, file2, fad_score))
    return results

def main():
    # Initialize FAD metric
    fad_calculator = FADMetric()
    
    # Takes two input genres (or instruments or moods or any sort of dataset) and compares the FAD scores between all songs within each genre and between the two genres
    genre1 = "metal"
    genre2 = "classical"

    # Define directories
    genre1_dir = "/Users/enikolak/Desktop/personal/musictherapy/our_implementation/musicgen-dreamboothing/tests/Data/genres_original/" + genre1 + "_shorter"
    genre2_dir = "/Users/enikolak/Desktop/personal/musictherapy/our_implementation/musicgen-dreamboothing/tests/Data/genres_original/" + genre2 + "_shorter"
    
    # Calculate FAD scores for genre1 songs
    print(f"Calculating FAD scores for {genre1} songs:")
    genre1_results = calculate_fad_for_directory(fad_calculator, genre1_dir)
    genre1_scores = [score for _, _, score in genre1_results]
    for file1, file2, score in genre1_results:
        print(f"FAD between {file1} and {file2}: {score}")
    print(f"\nAverage FAD score for {genre1}: {np.mean(genre1_scores):.4f}")
    
    # Calculate FAD scores for genre2 songs
    print(f"\nCalculating FAD scores for {genre2} songs:")
    genre2_results = calculate_fad_for_directory(fad_calculator, genre2_dir)
    genre2_scores = [score for _, _, score in genre2_results]
    for file1, file2, score in genre2_results:
        print(f"FAD between {file1} and {file2}: {score}")
    print(f"\nAverage FAD score for {genre2}: {np.mean(genre2_scores):.4f}")
    
    # Calculate FAD scores between genre1 and genre2 songs
    print(f"\nCalculating FAD scores between {genre1} and {genre2}:")
    cross_genre_results = calculate_cross_genre_fad(fad_calculator, genre1_dir, genre2_dir)
    cross_genre_scores = [score for _, _, score in cross_genre_results]
    for genre1_file, genre2_file, score in cross_genre_results:
        print(f"FAD between {genre1_file} and {genre2_file}: {score}")
    print(f"\nAverage FAD score between {genre1} and {genre2}: {np.mean(cross_genre_scores):.4f}")

    # Print overall summary
    print("\nSummary:")
    print(f"Average FAD score within {genre1}: {np.mean(genre1_scores):.4f}")
    print(f"Average FAD score within {genre2}: {np.mean(genre2_scores):.4f}")
    print(f"Average FAD score between {genre1} and {genre2}: {np.mean(cross_genre_scores):.4f}")

if __name__ == "__main__":
    main()