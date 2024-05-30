import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import torch
import random

def find_bed_and_fasta_files(folder_path):
    # Pattern to match BED files from dataset0.tss.bed to dataset4.tss.bed
    bed_pattern = os.path.join(folder_path, '*[0-4].tss.bed')
    # Patterns to find the FASTA files with .fasta or .fa extensions
    fasta_patterns = [os.path.join(folder_path, '*.fasta'), os.path.join(folder_path, '*.fa')]

    # Use glob to find files matching the BED pattern
    bed_files = glob.glob(bed_pattern)

    # Initialize an empty list to collect FASTA files found
    fasta_files = []
    # Search for FASTA files matching each pattern and extend the list
    for pattern in fasta_patterns:
        fasta_files.extend(glob.glob(pattern))

    # Ensure we found exactly one FASTA file
    if len(fasta_files) != 1:
        raise FileNotFoundError("Expected exactly one FASTA file, found {}: {}".format(len(fasta_files), fasta_files))
    fasta_file = fasta_files[0]

    return bed_files, fasta_file

# Function to read the genomic sequence from a FASTA file (simplified for one chromosome)
def read_genomic_sequence(fasta_file):
    with open(fasta_file, 'r') as f:
        next(f)  # Skip the header
        sequence = ''.join(line.strip() for line in f)
    return sequence

def plot_tss_distribution(window_start, window_end, tss_positions, genomic, folder_path, window_size=6000):
    """
    Plot the TSS distribution for a specified window and handle genomic nucleotides.
    """
    # Extract the genomic sequence for the current window
        # Constants
    upstream_len = 100
    tss_len = 100
    downstream_len = 100
    window_sequence = genomic[window_start:window_end]

    # Save the window sequence to a text file
    sequence_file_path = os.path.join(folder_path, 'plot', f'sequence_{window_start}-{window_start + window_size}.txt')
    with open(sequence_file_path, 'w') as seq_file:
        seq_file.write(window_sequence)

    # The rest of the function remains largely the same...
    # Initialize an array for the window range to accumulate probabilities
    prob_accumulator = np.zeros(window_size, dtype=float)
    local_tss_positions = tss_positions[(tss_positions >= window_start) & (tss_positions < window_end)]

    for tss_pos in local_tss_positions:
        local_range_start = max(tss_pos - upstream_len, window_start)
        local_range_end = min(tss_pos + downstream_len, window_end)
        local_range = np.arange(local_range_start, local_range_end)
        
        prob_dist = norm.pdf(local_range, loc=tss_pos, scale=tss_len/6)
        prob_dist /= np.max(prob_dist)
        
        indexes = local_range - window_start
        valid_indexes = (indexes >= 0) & (indexes < window_size)
        prob_accumulator[indexes[valid_indexes]] += prob_dist[valid_indexes]

    prob_accumulator = np.minimum(prob_accumulator, 1)

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(window_start, window_start + window_size), prob_accumulator, label='Accumulated TSS Probability')
    plt.fill_between(np.arange(window_start, window_start + window_size), 0, prob_accumulator, alpha=0.4)
    plt.title(f'TSS Probability Distribution: Positions {window_start} to {window_start + window_size}')
    plt.xlabel('Position (nucleotide)')
    plt.ylabel('Accumulated Probability')
    plt.legend()

    plot_folder = os.path.join(folder_path, 'plot')
    os.makedirs(plot_folder, exist_ok=True)
    plt.savefig(os.path.join(plot_folder, f'plot_{window_start}-{window_start + window_size}.png'))
    plt.close()

def sequence_to_onehot(seq):
    """Convert a genomic sequence to a one-hot encoded tensor."""
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    onehot_encoded = [mapping.get(nucleotide, [0, 0, 0, 0]) for nucleotide in seq]
    return torch.tensor(onehot_encoded, dtype=torch.float32)

def save_tss_data(window_start, window_end, tss_positions, genomic, folder_path, window_size=6000):
    """
    Save the TSS distribution and one-hot encoded genomic sequence for a specified window.
    """

    # Constants
    upstream_len = 100
    tss_len = 30
    downstream_len = 100
    # Extract the genomic sequence for the current window and convert it to one-hot encoding
    window_sequence = genomic[window_start:window_end]
    encoded_sequence = sequence_to_onehot(window_sequence)

    # Initialize an array for the window range to accumulate probabilities
    prob_accumulator = np.zeros(window_size, dtype=float)
    local_tss_positions = tss_positions[(tss_positions >= window_start) & (tss_positions < window_end)]

    for tss_pos in local_tss_positions:
        local_range_start = max(tss_pos - upstream_len, window_start)
        local_range_end = min(tss_pos + downstream_len, window_end)
        local_range = np.arange(local_range_start, local_range_end)
        
        prob_dist = norm.pdf(local_range, loc=tss_pos, scale=tss_len/6)
        prob_dist /= np.max(prob_dist)
        
        indexes = local_range - window_start
        valid_indexes = (indexes >= 0) & (indexes < window_size)
        prob_accumulator[indexes[valid_indexes]] += prob_dist[valid_indexes]

    prob_accumulator = np.minimum(prob_accumulator, 1)
    prob_accumulator_tensor = torch.tensor(prob_accumulator, dtype=torch.float32)

    # Save the one-hot encoded sequence and TSS probability distribution as a PyTorch tensor
    data_folder = os.path.join(folder_path, 'data')
    os.makedirs(data_folder, exist_ok=True)
    torch.save({'sequence': encoded_sequence, 'distribution': prob_accumulator_tensor},
               os.path.join(data_folder, f'data_{window_start}-{window_start + window_size}.pt'))


def save_tss_randomized_data_with_normal_distribution(tss_positions, genomic, folder_path, fold, mode, window_size=256, tss_len=30):
    """
    Generate and save TSS data with TSS positions randomized within a specified range in the window,
    using a normal distribution centered on the TSS position to represent TSS activity.
    """
    data = {}
    min_tss_pos = 40
    max_tss_pos = 216
    tss_len = 30

    for idx, tss_pos in enumerate(tss_positions):
        # Randomize TSS position within the specified range
        tss_offset = random.randint(min_tss_pos, max_tss_pos)
        
        # Calculate window start and end based on randomized TSS position
        window_start = max(0, tss_pos - tss_offset)
        window_end = min(len(genomic), window_start + window_size)
        
        if window_end > len(genomic):
            window_end = len(genomic)
            window_start = window_end - window_size

        # Extract sequence and encode
        window_sequence = genomic[window_start:window_end]
        encoded_sequence = sequence_to_onehot(window_sequence)
        encoded_sequence_tensor = torch.tensor(encoded_sequence, dtype=torch.float32)
        
        # Generate a normal distribution centered on the TSS within the window
        x_range = np.arange(0, window_size)
        tss_window_pos = tss_offset - (window_start - (tss_pos - tss_offset))
        prob_distribution = norm.pdf(x_range, loc=tss_window_pos, scale=tss_len / 6)
        prob_distribution /= np.max(prob_distribution)  # Normalize to have a maximum of 1
        prob_distribution_tensor = torch.tensor(prob_distribution, dtype=torch.float32)
        
        # Save to dictionary
        data[f'window_{tss_pos}'] = {'sequence': encoded_sequence_tensor, 'distribution': prob_distribution_tensor}
    
    # Save data
    data_folder = os.path.join(folder_path, 'data', f'fold_{fold}')
    os.makedirs(data_folder, exist_ok=True)
    torch.save(data, os.path.join(data_folder, f'{mode}_data.pt'))


def save_non_tss_data(genomic, folder_path, fold, mode, window_size=256, non_tss_count=1000):
    """
    Generate and save data windows from random genomic positions that are not known TSS positions.
    """
    data = {}
    genomic_length = len(genomic)

    # Generate random positions for non-TSS data
    non_tss_positions = set()
    while len(non_tss_positions) < non_tss_count:
        random_pos = random.randint(0, genomic_length - window_size)
        non_tss_positions.add(random_pos)

    for idx, pos in enumerate(non_tss_positions):
        window_sequence = genomic[pos:pos + window_size]
        encoded_sequence = sequence_to_onehot(window_sequence)
        encoded_sequence_tensor = torch.tensor(encoded_sequence, dtype=torch.float32)
        
        # The probability distribution for non-TSS data will be zeros
        prob_distribution_tensor = torch.zeros(window_size, dtype=torch.float32)

        # Save to dictionary
        data[f'non_tss_window_{pos}'] = {'sequence': encoded_sequence_tensor, 'distribution': prob_distribution_tensor}

    # Save data
    data_folder = os.path.join(folder_path, 'data', f'fold_{fold}')
    os.makedirs(data_folder, exist_ok=True)
    filename = f'non_tss_data.pt' if mode == "train" else f'non_tss_validation_data.pt'
    torch.save(data, os.path.join(data_folder, filename))


# Function to read TSS positions from BED files
def read_tss_positions(bed_files):
    if not isinstance(bed_files, list):
        bed_files = [bed_files]  # Convert to list if it's a single element
        
    all_positions = []
    for bed_file in bed_files:
        df = pd.read_csv(bed_file, sep='\t', header=None, usecols=[1], names=['start'])
        all_positions.append(df['start'].values)
    combined_positions = np.concatenate(all_positions)
    unique_positions = np.unique(combined_positions)
    return unique_positions

# Specify the folder containing your BED and FASTA files
folder_path = 'athaliana'

# Automatically find BED and FASTA files
bed_files, fasta_file = find_bed_and_fasta_files(folder_path)

# Read positions and genomic sequence
tss_positions = read_tss_positions(bed_files)
genomic = read_genomic_sequence(fasta_file)

print(bed_files)

# For each fold, create training and validation sets
for fold in range(5):
    # Copy the list to avoid modifying the original
    current_files = bed_files.copy()
    validation_file = current_files.pop(fold)  # Use the copy for operations

    print(validation_file)

    all_positions = read_tss_positions(bed_files)
    training_positions = read_tss_positions(validation_file)
    diff_val = np.setdiff1d(all_positions, training_positions, assume_unique=True)
    
    print(len(training_positions), len(diff_val))

    # Save data for training and validation
    save_tss_randomized_data_with_normal_distribution(training_positions, genomic, folder_path,  fold, "train", window_size=256)
    save_tss_randomized_data_with_normal_distribution(diff_val, genomic, folder_path,  fold, "validation", window_size=256)

    save_non_tss_data(genomic, folder_path, fold, "train", non_tss_count=5000)
    save_non_tss_data(genomic, folder_path, fold, "validation", non_tss_count=1000)  # Assuming you want fewer samples for validation