import pandas as pd
from Bio import SeqIO
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import matplotlib.pyplot as plt
from Bio import SeqIO
import re
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.lines as mlines
import torch.nn.functional as F
from sklearn.cluster import KMeans
from collections import Counter
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import warnings
import matplotlib.colors as mcolors
import argparse
from sklearn.model_selection import train_test_split
from prettytable import PrettyTable
from mamba_ssm import Mamba
from rotary_embedding_torch import RotaryEmbedding
#from rotary_embedding_torch import RotaryEmbedding
from utils import plot_residuals
from utils import plot_bland_altman
from utils import plot_accuracy
from utils import plot_error_histogram
from utils import scatter_plot
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="matplotlib")
import matplotlib        
import math
matplotlib.use('Agg')
torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True
from numpy.random import RandomState
import pickle
from scipy.stats import norm
import logging
from torch.utils.data import ConcatDataset
import glob
import pickle

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
import random
from einops.layers.torch import Rearrange
from modeling_enformer import Enformer, seq_indices_to_one_hot
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from stripedhyena.tokenizer import CharLevelTokenizer
from evo.scoring import prepare_batch

import re
from sklearn.model_selection import train_test_split

# Check GPU availability
batch_size = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
random.seed(42)

def read_fasta(file_path):
    """Read a FASTA file and return a list of tuples with header and sequence."""
    sequences = []
    header = None
    seq = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if header:
                    sequences.append((header, ''.join(seq)))
                header = line
                seq = []
            else:
                seq.append(line)  # Keep sequences intact, including TSS markers
        if header:
            sequences.append((header, ''.join(seq)))
    return sequences

def extract_tss_segments(sequences, window=2000):
    extracted_segments = []
    for header, seq in sequences:
        tss_matches = list(re.finditer(r'\*([ACGT]{1,500}?)\*', seq))

        for match in tss_matches:
            start_pos, end_pos = match.start(), match.end()
            upstream_start = max(0, start_pos - window)
            downstream_end = min(len(seq), end_pos + window)

            # Calculate entire_seq including the markers for the adjustment reference
            entire_seq_with_markers = seq[upstream_start:downstream_end]
            # Calculate the sequence without markers for actual processing
            entire_seq = entire_seq_with_markers.replace("*", "")

            # Adjustment for removed * symbols for TSS positions
            adjustment = [i for i, char in enumerate(entire_seq_with_markers) if char == '*']
            adjusted_tss_positions = []
            for tss_match in tss_matches:
                tss_start, tss_end = tss_match.start(), tss_match.end()
                # Count the number of * removed before each TSS start and end position for adjustment
                adjust_start = sum(1 for adj in adjustment if adj < tss_start - upstream_start)
                adjust_end = sum(1 for adj in adjustment if adj < tss_end - upstream_start)
                # Adjust the start and end positions
                adjusted_start = tss_start - upstream_start - adjust_start
                adjusted_end = tss_end - upstream_start - adjust_end - 1  # -1 for the end position adjustment
                adjusted_tss_positions.append((adjusted_start, adjusted_end))

            # Append the extracted segment with adjusted TSS positions
            extracted_segments.append((header, entire_seq, adjusted_tss_positions))
    return extracted_segments

class TSSDataset(Dataset):
    def __init__(self, tss_segments):
        self.tss_segments = tss_segments
        self.window_size = 1500
        self.mean = 750#1324.8877  # Mean value around which the TSS will be centralized
        self.std = 236.7499  # Standard deviation for centralization

    def __len__(self):
        return len(self.tss_segments)
    
    def sequence_to_onehot(self, seq):
        mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
        return [mapping.get(nucleotide, [0, 0, 0, 0]) for nucleotide in seq] 

    def __getitem__(self, idx):
        header, entire_seq, relative_tss_positions = self.tss_segments[idx]

        # Sample tss_centered_position within a reasonable range
        tss_centered_position = random.randint(100, 1450)#int(np.random.normal(self.mean, self.std, 1))
        attempt = 0  # Debug: Count sampling attempts
        while not (0 < tss_centered_position < self.window_size):
            tss_centered_position = int(np.random.normal(self.mean, self.std, 1))
            attempt += 1  # Debug: Increment attempt counter

        # Find the primary TSS and calculate its new centered position
        primary_tss_pos = min(relative_tss_positions, key=lambda x: abs(x[0]-2000))
        start_pos = max(0, min(primary_tss_pos[0] - tss_centered_position, len(entire_seq) - self.window_size))
        end_pos = start_pos + self.window_size

        # Crop the sequence to the desired window size
        cropped_seq = entire_seq[start_pos:end_pos]

        # Update the TSS positions to be relative to the start of the cropped sequence
        # and ensure they fall within the cropped sequence bounds
        updated_tss_positions = [(max(0, start - start_pos), min(end - start_pos, self.window_size - 1)) for start, end in relative_tss_positions if start_pos <= start < end_pos]

        # Constants
        MAX_TSS_WIDTH = 160  # Maximum width for TSS region to apply normal distribution
        ENFORCED_TSS_WIDTH = 160  # Enforce a width of 50 nucleotides for all TSS regions
        HALF_ENFORCED_WIDTH = ENFORCED_TSS_WIDTH // 2

        # Generate a normal distribution centered on each TSS
        x_range = np.arange(0, self.window_size)
        prob_dist = np.zeros_like(x_range, dtype=np.float32)

        for tss_start, tss_end in updated_tss_positions:
            tss_len = tss_end - tss_start + 1
            # Calculate the center of the TSS region
            tss_center = (tss_start + tss_end) / 2 + random.randint(-2, 2)

            if tss_len > MAX_TSS_WIDTH:
                # Center the normal distribution on the midpoint of the TSS
                effective_start = max(0, int(tss_center) - HALF_ENFORCED_WIDTH)
                effective_end = min(self.window_size, int(tss_center) + HALF_ENFORCED_WIDTH)
                effective_tss_len = ENFORCED_TSS_WIDTH
            else:
                effective_start = tss_start
                effective_end = tss_end
                effective_tss_len = tss_len

            # Calculate the scale factor based on the effective TSS length
            scale_factor = effective_tss_len / 6

            # Generate the normal distribution for the effective TSS range
            tss_prob_dist = norm.pdf(x_range, loc=tss_center, scale=scale_factor)
            tss_prob_dist /= tss_prob_dist.max()  # Normalize to have a maximum of 1

            # Adjust probabilities for non-TSS regions to approach zero
            tss_prob_dist[:effective_start] = tss_prob_dist[:effective_start] * 0.01  # Dampen upstream
            tss_prob_dist[effective_end + 1:] = tss_prob_dist[effective_end + 1:] * 0.01  # Dampen downstream

            prob_dist = np.maximum(prob_dist, tss_prob_dist) #''' # Combine distributions, taking the maximum at each position

        prob_dist_tensor = torch.tensor(prob_dist, dtype=torch.float32)

        onehot_seq = self.sequence_to_onehot(cropped_seq)
        onehot_seq_tensor = torch.tensor(onehot_seq, dtype=torch.float32).view(self.window_size, 4)

        return header, onehot_seq_tensor, prob_dist_tensor#, updated_tss_positions

class TSSDatasetValidation(Dataset):
    def __init__(self):
        self.directory_path = 'experiment_arab/'
        self.window_size = 1500

        # List all text files in the directory
        file_paths = glob.glob(os.path.join(self.directory_path, '*.txt'))

        self.sequences = []
        self.position_tss = []
        
        # Process files
        for file_path in tqdm(file_paths, desc="Processing Files"):
            sequence = self.read_sequence_from_file(file_path)
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            match = re.search(r'GT_(\d+)', file_name)
            gt_number = int(match.group(1)) if match else None

            self.sequences.append(sequence)
            self.position_tss.append(gt_number)

    def read_sequence_from_file(self, file_path):
        with open(file_path, 'r') as file:
            sequence = file.read().strip()
        return sequence

    def __len__(self):
        return len(self.sequences)
    
    def sequence_to_onehot(self, seq):
        mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
        return [mapping.get(nucleotide, [0, 0, 0, 0]) for nucleotide in seq] 

    def __getitem__(self, idx):
        seq = self.sequences[idx]  # Get the sequence for the current index
        onehot_seq = self.sequence_to_onehot(seq)  # Convert the sequence to one-hot encoding
        onehot_seq_tensor = torch.tensor(onehot_seq, dtype=torch.float32).view(self.window_size, 4)  # Reshape
        gt_position = self.position_tss[idx]  # Get the ground truth position for the current index
        
        # Creating a normal distribution centered at gt_position with size 25
        x_range = np.arange(0, self.window_size)
        prob_dist = np.zeros_like(x_range, dtype=np.float32)
        if gt_position is not None:
            spread = 25  # Define the spread of the normal distribution
            scale = spread / 6  # Approximation to make the significant area cover +/- 25 positions
            prob_dist = norm.pdf(x_range, loc=gt_position, scale=scale)
            prob_dist /= prob_dist.max()  # Normalize to have a maximum of 1

        prob_dist_tensor = torch.tensor(prob_dist, dtype=torch.float32)
        
        return " ", onehot_seq_tensor, prob_dist_tensor

parser = argparse.ArgumentParser(description='Train on specific dataset')
parser.add_argument('--dataset', type=str, choices=['arabidopsis', 'maize', 'both'], required=True,
                    help='Choose the dataset to train on: arabidopsis, maize, or both.')
args = parser.parse_args()


def generate_subsequences(sequence, window_size=2000):
    """Generate all possible subsequences of a given length from the sequence."""
    return set(sequence[i:i+window_size] for i in range(len(sequence) - window_size + 1))

def check_for_subsequence_overlap(train_data, val_data, window_size=2000):
    """Check for overlapping subsequences between training and validation datasets."""
    train_subsequences = set()
    val_subsequences = set()

    for _, seq, _ in train_data:
        train_subsequences.update(generate_subsequences(seq, window_size))
    
    for _, seq, _ in val_data:
        val_subsequences.update(generate_subsequences(seq, window_size))
    
    overlap = train_subsequences.intersection(val_subsequences)
    return overlap

def remove_overlapping_subsequences(validation_data, overlap):
    """Remove sequences from the validation dataset that contain any overlapping subsequences."""
    filtered_validation_data = []
    for header, seq, positions in validation_data:
        # Generate all possible subsequences for the current sequence
        seq_subsequences = generate_subsequences(seq, window_size=200)
        
        # Check if the current sequence contains any overlapping subsequences
        if not seq_subsequences.intersection(overlap):
            # If no overlap, include the sequence in the filtered validation dataset
            filtered_validation_data.append((header, seq, positions))
    
    return filtered_validation_data

if args.dataset == 'arabidopsis':
    '''sequences = read_fasta("Arabidopsis_TAIR10_PEAT_TSS_allPeak_High_quality_marks.fa")
    tss_segments = extract_tss_segments(sequences)

    # Calculate the split index
    split_index = int(len(tss_segments) * 0.9498)

    # Split the tss_segments into training and validation sets
    training_tss_segments = tss_segments[:split_index]
    validation_tss_segments = tss_segments[split_index:]

    overlap = check_for_subsequence_overlap(training_tss_segments, validation_tss_segments, window_size=200)
    # Assuming overlap has been identified as above
    if overlap:
        print(f"Found {len(overlap)} overlapping subsequences. Removing from validation dataset.")
        validation_tss_segments = remove_overlapping_subsequences(validation_tss_segments, overlap)
        print(f"Validation dataset size after removal: {len(validation_tss_segments)}")
    else:
        print("No overlapping subsequences found.")'''
    
    # Load validation_tss_segments
    with open('validation_tss_segments_arabidopsis.pkl', 'rb') as f:
        validation_tss_segments = pickle.load(f)

    with open('training_tss_segments_arabidopsis.pkl', 'rb') as f:
        training_tss_segments = pickle.load(f)

    #print(f"Total TSS segments: {len(tss_segments)}")
    print(f"Training TSS segments: {len(training_tss_segments)}")
    print(f"Validation TSS segments: {len(validation_tss_segments)}")

    '''# Save training_tss_segments
    with open('training_tss_segments.pkl', 'wb') as f:
        pickle.dump(training_tss_segments, f)

    # Save validation_tss_segments
    with open('validation_tss_segments.pkl', 'wb') as f:
        pickle.dump(validation_tss_segments, f)'''

    training_dataset = TSSDataset(training_tss_segments)
    validation_dataset = TSSDataset(validation_tss_segments)
    #validation_dataset =  TSSDatasetValidation()

    train_loader_tss = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader_tss = DataLoader(validation_dataset, batch_size=64, shuffle=False, num_workers=4)
if args.dataset == 'maize':
    '''sequences = read_fasta("Maize_AGPv4_mark_TSS_highly_stringent_7reps.fa")
    tss_segments = extract_tss_segments(sequences)

    # Calculate the split index
    split_index = int(len(tss_segments) * 0.9498)

    # Split the tss_segments into training and validation sets
    training_tss_segments = tss_segments[:split_index]
    validation_tss_segments = tss_segments[split_index:]'''

    '''overlap = check_for_subsequence_overlap(training_tss_segments, validation_tss_segments, window_size=200)
    # Assuming overlap has been identified as above
    if overlap:
        print(f"Found {len(overlap)} overlapping subsequences. Removing from validation dataset.")
        validation_tss_segments = remove_overlapping_subsequences(validation_tss_segments, overlap)
        print(f"Validation dataset size after removal: {len(validation_tss_segments)}")
    else:
        print("No overlapping subsequences found.")'''
    
    # Save training_tss_segments
    '''with open('training_tss_segments_maize_highly_stringent_7reps.pkl', 'wb') as f:
        pickle.dump(training_tss_segments, f)

    # Save validation_tss_segments
    with open('validation_tss_segments_maize_highly_stringent_7reps.pkl', 'wb') as f:
        pickle.dump(validation_tss_segments, f)'''


    # Load validation_tss_segments
    with open('validation_tss_segments_maize_highly_stringent_7reps.pkl', 'rb') as f:
        validation_tss_segments = pickle.load(f)

    with open('training_tss_segments_maize_highly_stringent_7reps.pkl', 'rb') as f:
        training_tss_segments = pickle.load(f)

    #print(f"Total TSS segments: {len(tss_segments)}")
    print(f"Training TSS segments: {len(training_tss_segments)}")
    print(f"Validation TSS segments: {len(validation_tss_segments)}")

    training_dataset = TSSDataset(training_tss_segments)
    validation_dataset = TSSDataset(validation_tss_segments)
    #validation_dataset =  TSSDatasetValidation()

    train_loader_tss = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader_tss = DataLoader(validation_dataset, batch_size=64, shuffle=False, num_workers=4)


'''class CombinedModel(torch.nn.Module):
    def __init__(self, target_length=256, dropout=0.01):
        super().__init__()

        self.enformer  = Enformer.from_hparams(
            dim = 1536 // 4,
            depth = 4,
            heads = 8,
            use_tf_gamma = False,
            output_heads = dict(prob = 1500),
            target_length = 1500,
        )
        
    def forward(self, one_hot_sequence):
        # Get the output and embeddings from the enformer
        output = self.enformer(one_hot_sequence)
        classification = output['prob'].mean(dim=1)
        classification_clamped = torch.clamp(classification, 0, 1)

        return classification_clamped'''
        
class CombinedModel(torch.nn.Module):
    def __init__(self, target_length=256, dropout=0.01):
        super().__init__()
        # Global model
        self.enformer_global = Enformer.from_hparams(
            dim = 1536 // 4,
            depth = 4,
            heads = 8,
            use_tf_gamma = False,
            output_heads = dict(prob = 1500),
            target_length = 1500,
        )
        # Local model
        self.enformer_local = Enformer.from_hparams(
            dim = 1536 // 4,
            depth = 4,
            heads = 8,
            use_tf_gamma = False,
            output_heads = dict(p1 = 250, p2 = 250, p3 = 250, p4 = 250, p5 = 250, p6 = 250),
            target_length = 250,
        )
        
    def forward(self, one_hot_sequence):
        # Global predictions
        global_output = self.enformer_global(one_hot_sequence)
        classification_global = global_output['prob'].mean(dim=1)
        
        # Local predictions - split the input sequence and apply the local model to each part
        splits = torch.split(one_hot_sequence, 250, dim=-2)  # Assuming the last dimension is the sequence length
        local_outputs = []
        for i, split in enumerate(splits):
            output = self.enformer_local(split)
            local_outputs.append(output[f'p{i+1}'].mean(dim=1))  # Collect each segment's output
        
        # Concatenate local outputs to recreate a sequence of 1500 bases
        combined_local_outputs = torch.cat(local_outputs, dim=-1)  # Concatenate along the sequence length dimension
        # Combine global and local outputs - here we simply add them, but consider other methods depending on your model's needs
        #combined_output = classification_global + combined_local_outputs  # Element-wise addition
        
        classification_clamped_local = torch.clamp(combined_local_outputs, 0, 1)
        classification_clamped_global = torch.clamp(classification_global, 0, 1)

        return classification_clamped_global, classification_clamped_local
    
def plot_example(preds_numpy, labels_numpy, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(preds_numpy, label='Predictions', color='blue', marker='o')
    plt.plot(labels_numpy, label='True Labels', color='red', linestyle='--')
    plt.title('Prediction vs True Label')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def save_output_to_file(epoch, tolerance, accuracy, dataset):
    # Construct the file name
    file_name = f"{dataset}.txt"
    
    # Create or open the file in append mode
    with open(file_name, 'a') as f:
        # Write the output to the file
        f.write(f"Epoch {epoch + 1} Overall TSS Accuracy within {tolerance} nt: {accuracy:.4f}\n")

lr = 0.0005
wd = 0.0001
tolerance_values = list(range(1, 12))
combined_model = CombinedModel(device).to(device)

model_path = 'best_model_based_on_pearson.pth'

# Load the pre-trained model state dictionary
#state_dict = torch.load(model_path, map_location=device)

# Load the filtered state dictionary, set strict=False to ignore missing keys
#combined_model.load_state_dict(state_dict)

# Wrap the model with DataParallel for multi-GPU training
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    combined_model = nn.DataParallel(combined_model)

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def poisson_loss(pred, target):
    return (pred - target * log(pred)).mean()

def pearson_corr_coef(x, y, dim = 1, reduce_dims = (-1,)):
    x_centered = x - x.mean(dim = dim, keepdim = True)
    y_centered = y - y.mean(dim = dim, keepdim = True)
    return F.cosine_similarity(x_centered, y_centered, dim = dim).mean(dim = reduce_dims)

optimizer = torch.optim.AdamW(combined_model.parameters(), lr=lr, weight_decay=wd)
criterion = poisson_loss
best_val_loss = float('inf')
best_val_pearson_corr = -float('inf')  # Initialize with the worst possible correlation
best_sum = 0
#0.6035

threshold = 0.01  # Define a threshold for TSS identification, adjust based on your distribution

# Validation phase
combined_model.eval()
val_loss = 0.0
# Aggregate Pearson correlation coefficients for validation
val_corr_coefs = []
all_predictions = []  # Store all predictions as tensors
all_labels = []  # Store all labels as tensors

with torch.no_grad():
    for header, cropped_seq, prob_dist_tensor in tqdm(val_loader_tss, desc="Validating"):
        prob_dist_tensor = prob_dist_tensor.to(device)
        classification_clamped_global, classification_clamped_local = combined_model(cropped_seq.to(device))
        pred = classification_clamped_global + classification_clamped_local
        loss = criterion(pred, prob_dist_tensor)

        max_pred_index_predictions = torch.argmax(pred, dim=1)  # Finds the argmax for each item in the batch along the classes dimension
        max_pred_index_labels = torch.argmax(prob_dist_tensor, dim=1)  # Similar for the ground truth labels
        
        # Accumulate predictions and labels tensors for later analysis
        all_predictions.append(max_pred_index_predictions)
        all_labels.append(max_pred_index_labels)

        val_loss += loss.item()
        # Assuming pearson_corr_coef is defined correctly and can handle the inputs
        corr_coef = pearson_corr_coef(pred, prob_dist_tensor).item()
        val_corr_coefs.append(corr_coef)

# Concatenate all batched predictions and labels into single tensors
all_predictions_tensor = torch.cat(all_predictions, dim=0)
all_labels_tensor = torch.cat(all_labels, dim=0)

# Convert the concatenated tensors to NumPy arrays for easier processing
all_predictions_np = all_predictions_tensor.cpu().numpy()
all_labels_np = all_labels_tensor.cpu().numpy()

diffs = np.abs(all_predictions_np - all_labels_np)  # Calculate the absolute differences


numbers = list(range(1, 31))
specified_tolerances = numbers
# Calculate and print overall mean accuracies
#specified_tolerances = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]
comparison = pd.DataFrame(diffs, columns=["position_diff"])
overall_mean_accuracies = {}

total_predictions = len(comparison)
for tolerance in specified_tolerances:
    matches_within_tolerance = comparison[comparison["position_diff"] <= tolerance].shape[0]
    overall_mean_accuracies[tolerance] = matches_within_tolerance / total_predictions

accuracy_table = PrettyTable()
accuracy_table.field_names = ["Tolerance", "Mean Accuracy"]
mean_accuracys = []
for tolerance, mean_accuracy in overall_mean_accuracies.items():
    accuracy_table.add_row([tolerance, f"{mean_accuracy:.4f}"])
    mean_accuracys.append(mean_accuracy)

print(accuracy_table)
current_sum_accuracy = sum(mean_accuracys)
print("current_sum_accuracy ", current_sum_accuracy)
'''
with torch.no_grad():
    for i, (header, cropped_seq, prob_dist_tensor) in enumerate(tqdm(val_loader_tss, desc="Validating")):
        prob_dist_np = prob_dist_tensor[0].cpu().numpy()  # Convert the first tensor to numpy
        
        plt.figure(figsize=(20, 5))
        plt.plot(prob_dist_np, label='Probability Distribution', color='blue')

        # Identify TSS regions
        tss_regions = []
        in_region = False
        for pos, prob in enumerate(prob_dist_np):
            if prob > threshold and not in_region:
                # Start of a new TSS region
                start_pos = pos
                in_region = True
            elif prob <= threshold and in_region:
                # End of a TSS region
                end_pos = pos
                tss_regions.append((start_pos, end_pos))
                in_region = False
        
        # If we end still in a region, close it
        if in_region:
            tss_regions.append((start_pos, len(prob_dist_np)))

        # Plot and annotate TSS regions
        for start_pos, end_pos in tss_regions:
            plt.axvline(x=start_pos, color='green', linestyle='--', lw=2)
            plt.axvline(x=end_pos, color='red', linestyle='--', lw=2)
            region_size = end_pos - start_pos
            plt.text((start_pos+end_pos)/2, max(prob_dist_np), f'Size: {region_size}', color='black', fontsize=8, ha='center')

        plt.title(f'Probability Distribution Tensor for the First Sequence in Batch {i+1}')
        plt.xlabel('Position in Sequence')
        plt.ylabel('Probability')
        plt.legend()
        plt.savefig(f'validation/prob_dist_tensor_batch_{i+1}.png')
        plt.close()
'''


# Open a log file
log_file_path = "training_log.txt"
if not os.path.exists(log_file_path):
    with open(log_file_path, 'w') as log_file:
        log_file.write("Epoch,Current Sum Accuracy,Avg Corr Coef\n")

all_acc = 0
all_person = 0
for epoch in range(100):
    combined_model.train()
    running_loss = 0.0
    for header, cropped_seq, prob_dist_tensor in tqdm(train_loader_tss, desc=f"Epoch {epoch+1}/{50000}"):
        optimizer.zero_grad()

        # Assuming 'device' is defined (e.g., 'cuda' or 'cpu')
        classification_clamped_global, classification_clamped_local = combined_model(cropped_seq.to(device))
        #pred = classification_clamped_global + classification_clamped_local 
        loss_global  = criterion(classification_clamped_global, prob_dist_tensor.to(device))
        loss_local  = criterion(classification_clamped_local, prob_dist_tensor.to(device))
        loss = loss_global + loss_local
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader_tss)
    print(f"Training Loss: {epoch_loss:.4f}")

    # Validation phase
    combined_model.eval()
    val_loss = 0.0
    # Aggregate Pearson correlation coefficients for validation
    val_corr_coefs = []
    all_predictions = []  # Store all predictions as tensors
    all_labels = []  # Store all labels as tensors

    with torch.no_grad():
        for header, cropped_seq, prob_dist_tensor in tqdm(val_loader_tss, desc="Validating"):
            prob_dist_tensor = prob_dist_tensor.to(device)
            classification_clamped_global, classification_clamped_local  = combined_model(cropped_seq.to(device))
            
            loss_global  = criterion(classification_clamped_global, prob_dist_tensor.to(device))
            loss_local  = criterion(classification_clamped_local, prob_dist_tensor.to(device))
            loss = loss_global + loss_local
            pred = classification_clamped_global + classification_clamped_local
            #loss = criterion(pred, prob_dist_tensor)

            max_pred_index_predictions = torch.argmax(pred, dim=1)  # Finds the argmax for each item in the batch along the classes dimension
            max_pred_index_labels = torch.argmax(prob_dist_tensor, dim=1)  # Similar for the ground truth labels
            
            # Accumulate predictions and labels tensors for later analysis
            all_predictions.append(max_pred_index_predictions)
            all_labels.append(max_pred_index_labels)

            val_loss += loss.item()
            # Assuming pearson_corr_coef is defined correctly and can handle the inputs
            corr_coef = pearson_corr_coef(pred, prob_dist_tensor).item()
            val_corr_coefs.append(corr_coef)

    # Concatenate all batched predictions and labels into single tensors
    all_predictions_tensor = torch.cat(all_predictions, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)

    # Convert the concatenated tensors to NumPy arrays for easier processing
    all_predictions_np = all_predictions_tensor.cpu().numpy()
    all_labels_np = all_labels_tensor.cpu().numpy()

    diffs = np.abs(all_predictions_np - all_labels_np)  # Calculate the absolute differences

    numbers = list(range(1, 31))
    specified_tolerances = numbers
    # Calculate and print overall mean accuracies
    #specified_tolerances = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]
    comparison = pd.DataFrame(diffs, columns=["position_diff"])
    overall_mean_accuracies = {}

    total_predictions = len(comparison)
    for tolerance in specified_tolerances:
        matches_within_tolerance = comparison[comparison["position_diff"] <= tolerance].shape[0]
        overall_mean_accuracies[tolerance] = matches_within_tolerance / total_predictions

    accuracy_table = PrettyTable()
    accuracy_table.field_names = ["Tolerance", "Mean Accuracy"]
    mean_accuracys = []
    for tolerance, mean_accuracy in overall_mean_accuracies.items():
        accuracy_table.add_row([tolerance, f"{mean_accuracy:.4f}"])
        mean_accuracys.append(mean_accuracy)

    current_sum_accuracy = sum(mean_accuracys)
    all_acc+= current_sum_accuracy
    print("current_sum_accuracy ", current_sum_accuracy)

    val_loss /= len(val_loader_tss)
    avg_corr_coef = sum(val_corr_coefs) / len(val_corr_coefs)
    all_person+=avg_corr_coef
    print(f"Validation Loss: {val_loss:.4f}, Avg Pearson Corr Coef: {avg_corr_coef:.4f}")

    # At the end of each epoch, append the log with the new data
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"{epoch+1},{current_sum_accuracy:.4f},{avg_corr_coef:.4f}\n")
    
    # Save the model if the average Pearson correlation coefficient improved
    if avg_corr_coef > best_val_pearson_corr:
        print(f"Avg Pearson Corr Coef improved ({best_val_pearson_corr:.4f} --> {avg_corr_coef:.4f}). Saving model...")
        best_val_pearson_corr = avg_corr_coef
        #torch.save(combined_model.state_dict(), 'best_model_based_on_pearson.pth')
    
        # Select just one example from the last batch for plotting
        # Assuming the first dimension of `pred` and `prob_dist_tensor` is the batch dimension
        last_pred_single = pred[-1].cpu().numpy()  # Select the last example from the batch
        last_true_single = prob_dist_tensor[-1].cpu().numpy()

        plt.figure(figsize=(10, 5))
        plt.plot(last_pred_single.flatten(), label='Prediction', color='blue')
        plt.plot(last_true_single.flatten(), label='True Distribution', color='red', linestyle='--')
        plt.title('Comparison of the Last Validation Prediction and True Distribution for a Single Example')
        plt.xlabel('Sequence Position')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig('plot_tss_single_example.png')
        plt.close()
    # Save the model if the average Pearson correlation coefficient improved
    if current_sum_accuracy > best_sum:
        print(accuracy_table)
        print("SAVING ACC: ", current_sum_accuracy)
        best_sum = current_sum_accuracy
        torch.save(combined_model.state_dict(), 'best_model_based_on_SUM.pth')
    
        # Select just one example from the last batch for plotting
        # Assuming the first dimension of `pred` and `prob_dist_tensor` is the batch dimension
        last_pred_single = pred[-1].cpu().numpy()  # Select the last example from the batch
        last_true_single = prob_dist_tensor[-1].cpu().numpy()

        plt.figure(figsize=(10, 5))
        plt.plot(last_pred_single.flatten(), label='Prediction', color='blue')
        plt.plot(last_true_single.flatten(), label='True Distribution', color='red', linestyle='--')
        plt.title('Comparison of the Last Validation Prediction and True Distribution for a Single Example')
        plt.xlabel('Sequence Position')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig('plot_tss_single_example_best_sum.png')
        plt.close()

print("all_acc = ", all_acc)
print("all_person", all_person)