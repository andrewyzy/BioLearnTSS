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
import glob
from prettytable import PrettyTable

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
from scipy.ndimage import gaussian_filter1d


# Check GPU availability
batch_size = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
tokenizer = CharLevelTokenizer(512)
# Create a mapping of nucleotide to integer
nucleotide_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}

def convert_sequence_to_indices(sequence, nucleotide_to_index):
    return [nucleotide_to_index[nt] for nt in sequence if nt in nucleotide_to_index]

def generate_random_sequence(length):
    nucleotides = ['A', 'T', 'C', 'G']
    return ''.join(np.random.choice(nucleotides) for _ in range(length))

def pad_sequence(seq, max_padding, beginning=True):
    padding_length = np.random.randint(max_padding)
    random_seq = generate_random_sequence(padding_length)
    padded_seq = (random_seq + seq) if beginning else (seq + random_seq)
    return padded_seq, padding_length

def parse_fasta(file, origin):
    sequences = list(SeqIO.parse(file, "fasta"))
    print(f"Number of sequences in {os.path.basename(file)}: {len(sequences)}")
    return {str(record.id): str(record.seq) for record in sequences}, {str(record.id): origin for record in sequences}

class TSSDataset(Dataset):
    def __init__(self, fasta_file, seq_type, noise_rate=0.05, is_training=True):
        self.sequences = {}
        self.noise_rate = noise_rate
        self.is_training = is_training

        # Initialize max_len and origin
        self.max_len = 0
        self.origin = {}

        with open(fasta_file, 'r') as f:
            seq_id = ''
            seq_data = ''
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if seq_id:
                        self.sequences[seq_id] = seq_data
                        self.max_len = max(self.max_len, len(seq_data))
                        
                        # Store the entire unique ID as origin information
                        self.origin[seq_id] = seq_type
                        
                    seq_id = line[1:]
                    seq_data = ''
                else:
                    seq_data += line
            if seq_id:
                self.sequences[seq_id] = seq_data
                self.max_len = max(self.max_len, len(seq_data))

                # Store the entire unique ID as origin information
                self.origin[seq_id] = seq_type

        self.tss_ids = list(self.sequences.keys())


        for key in list(self.sequences.keys())[:5]:  
            print(key, self.sequences[key])
            
        np.random.shuffle(self.tss_ids)

    def get_tss_lengths(self):

        tss_lengths = [len(tss) for tss in self.tss_seqs.values()]
        return tss_lengths

    def cluster_tss(self):

        tss_lengths = self.get_tss_lengths()
        kmeans = KMeans(n_clusters=3, random_state=0).fit(np.array(tss_lengths).reshape(-1, 1))

        return kmeans.labels_

    def __len__(self):
        return len(self.tss_ids)  

    def sequence_to_onehot(self, seq):
        mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
        return [mapping.get(nucleotide, [0, 0, 0, 0]) for nucleotide in seq] 

    def introduce_noise(self, seq):
        nucleotides = np.array(['A', 'T', 'C', 'G'])

        num_noisy_positions = np.random.binomial(len(seq), self.noise_rate)

        noisy_indices = []
        if num_noisy_positions > 0:
            noisy_indices = np.random.choice(len(seq), size=num_noisy_positions, replace=False)
            random_nucleotides = np.random.choice(nucleotides, size=num_noisy_positions)
            seq_list = list(seq)
            for idx, nucl in zip(noisy_indices, random_nucleotides):
                seq_list[idx] = nucl

            seq = ''.join(seq_list)
        return seq, noisy_indices

    def generate_random_color(self):
        return np.random.rand(3,)

    def generate_random_sequence(self, length):
        nucleotides = ['A', 'T', 'C', 'G']
        sequence = ''.join(np.random.choice(nucleotides) for _ in range(length))
        indices = list(range(length))  
        return sequence, indices

    def __getitem__(self, idx):
        tss_id = self.tss_ids[idx]
        full_seq = self.sequences[tss_id]

        # Split the full sequence into downstream, tss, and upstream
        downstream, tss, upstream = full_seq.split('|')

        if self.is_training:
            downstream, _ = self.introduce_noise(downstream)
            upstream, _ = self.introduce_noise(upstream)
            tss, _ = self.introduce_noise(tss)

        # Process TSS if it is longer than a specified maximum length
        MAX_TSS_LEN = 160  # Maximum length for TSS
        if len(tss) > MAX_TSS_LEN:
            max_start_idx = len(tss) - MAX_TSS_LEN
            start_idx = random.randint(0, max_start_idx)
            tss = tss[start_idx:start_idx + MAX_TSS_LEN]
            #print(f"TSS length > 200, trimmed TSS to: {len(tss)}")

        MIN_SEQ_LEN = 40  # Minimum length for upstream and downstream
        tss_half_len = len(tss) // 2

        # Calculate the minimum and maximum center positions for the TSS
        low_val = MIN_SEQ_LEN + tss_half_len
        high_val = 256 - MIN_SEQ_LEN - tss_half_len

        # Randomly select a center position for TSS within the allowed range
        tss_center_in_combined_seq = np.random.randint(low_val, high_val + 1)

        # Calculate the target lengths for upstream and downstream
        upstream_target_length = tss_center_in_combined_seq - tss_half_len
        downstream_target_length = 256 - len(tss) - upstream_target_length

        # Adjust upstream and downstream sequences
        upstream = upstream[-upstream_target_length:]
        downstream = downstream[:downstream_target_length]

        combined_seq = upstream + tss + downstream

        upstream_len = len(upstream)
        tss_len = len(tss)
        downstream_len = len(downstream)
        
        # Generate a normal distribution centered on the TSS
        tss_start = upstream_len
        tss_end = upstream_len + tss_len
        x_range = np.arange(0, 256)  # Assuming combined sequence length is 256
        prob_dist = norm.pdf(x_range, loc=(tss_start + tss_end) / 2, scale=tss_len / 6)
        prob_dist /= prob_dist.max()  # Normalize to have a maximum of 1
        
        # Adjust probabilities for non-TSS regions to approach zero
        prob_dist[:tss_start] = prob_dist[:tss_start] * 0.01  # Dampen upstream
        prob_dist[tss_end:] = prob_dist[tss_end:] * 0.01  # Dampen downstream
        prob_dist_tensor = torch.tensor(prob_dist, dtype=torch.float32)


        seq_indices_tss = convert_sequence_to_indices(combined_seq, nucleotide_to_index)
        seq_indices_tensor = torch.tensor(seq_indices_tss, dtype=torch.long)

        if len(combined_seq) != 256:
            print("Sequence Length Mismatch!")
            print(f"Expected: 256, Got: {len(combined_seq)}")
            print(f"Upstream Length: {len(upstream)}")
            print(f"TSS Length: {len(tss)}")
            print(f"Downstream Length: {len(downstream)}")
            print(f"Total Combined Length: {len(upstream) + len(tss) + len(downstream)}")
            exit()  

        onehot_seq = self.sequence_to_onehot(combined_seq)
        onehot_seq_tensor = torch.tensor(onehot_seq, dtype=torch.float32).view(256, 4)
        tss_center_in_combined_seq = len(upstream) + len(tss) // 2
        
        return onehot_seq_tensor, tss_center_in_combined_seq, 1, self.origin[tss_id], seq_indices_tensor, prob_dist_tensor

def parse_control_fasta(file):
    sequences = list(SeqIO.parse(file, "fasta"))
    print(f"Number of control sequences in {os.path.basename(file)}: {len(sequences)}")
    return [str(record.seq) for record in sequences]

class ControlDataset(Dataset):
    def __init__(self, control_seqs, max_len, noise_rate=0.001, is_training=True, target_len=None, seed=None):
        self.control_seqs = control_seqs
        self.max_len = max_len
        self.noise_rate = noise_rate
        self.is_training = is_training
        self.target_len = target_len if target_len else len(control_seqs)
        self.nucleotides = ['A', 'T', 'C', 'G']
        self.random_state = RandomState(seed)

    def __len__(self):
        return self.target_len

    def sequence_to_onehot(self, seq):
        mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
        return [mapping.get(nucleotide, [0, 0, 0, 0]) for nucleotide in seq] 

    def augment_sequence(self, seq):

        choice = self.random_state.choice(5)

        if choice == 0:
            num_mutations = self.random_state.randint(1, 5)
            for _ in range(num_mutations):
                idx = self.random_state.randint(len(seq))
                mutation = self.random_state.choice(self.nucleotides)
                seq = seq[:idx] + mutation + seq[idx+1:]

        elif choice == 1:
            seq = seq[::-1]

        elif choice == 2:
            seq = ''.join(self.random_state.permutation(list(seq)))

        elif choice == 3 and len(seq) < 256:
            idx = self.random_state.randint(len(seq))
            insertion = self.random_state.choice(self.nucleotides)
            seq = seq[:idx] + insertion + seq[idx:]

        elif choice == 4 and len(seq) > 1:
            idx = self.random_state.randint(len(seq))
            seq = seq[:idx] + seq[idx+1:]

        # Ensure sequence is exactly 256 in length
        if len(seq) > 256:
            seq = seq[:256]
        elif len(seq) < 256:
            additional_nucleotides = ''.join(self.random_state.choice(self.nucleotides, 256 - len(seq), replace=True))
            seq += additional_nucleotides

        return seq

    def __getitem__(self, idx):

        if idx >= len(self.control_seqs):
            original_seq = self.control_seqs[self.random_state.randint(len(self.control_seqs))]
            control_seq = self.augment_sequence(original_seq)
        else:
            control_seq = self.control_seqs[idx]

        if len(control_seq) > 256:
            start_idx = np.random.randint(0, len(control_seq) - 256)
            control_seq = control_seq[start_idx:start_idx+256]

        #control_seq = self.augment_sequence(control_seq)

        if len(control_seq) != 256:
            print(f"Control Sequence Length Mismatch! Expected: 256, Got: {len(control_seq)}")
            exit()  

        seq_indices = convert_sequence_to_indices(control_seq, nucleotide_to_index)
        seq_indices_tensor = torch.tensor(seq_indices, dtype=torch.long)

        onehot_seq = self.sequence_to_onehot(control_seq)
        onehot_seq_tensor = torch.tensor(onehot_seq, dtype=torch.float32).view(256, 4)

        return onehot_seq_tensor, 0, 0, "control" , seq_indices_tensor, torch.zeros(256, dtype=torch.float32) 


parser = argparse.ArgumentParser(description='Train on specific dataset')
parser.add_argument('--dataset', type=str, choices=['arabidopsis', 'maize', 'both'], required=True,
                    help='Choose the dataset to train on: arabidopsis, maize, or both.')
args = parser.parse_args()

ArabTSSv2_train = "dataset/ArabTSSv2/ArabTSSv2_train.fasta"
ArabTSSv2_val = "dataset/ArabTSSv2/ArabTSSv2_val.fasta"
ArabTSSv2_test = "dataset/ArabTSSv2/ArabTSSv2_test.fasta"

MaizeTSSv4_train = "dataset/MaizeTSSv4/MaizeTSSv4_train.fasta"
MaizeTSSv4_val = "dataset/MaizeTSSv4/MaizeTSSv4_val.fasta"
MaizeTSSv4_test = "dataset/MaizeTSSv4/MaizeTSSv4_test.fasta"

ArabTSSv2_control_train = "dataset/ArabTSSv2/train_control_arabidopsis.fasta"
ArabTSSv2_control_val = "dataset/ArabTSSv2/val_control_arabidopsis.fasta"
ArabTSSv2_control_test = "dataset/ArabTSSv2/test_control_arabidopsis.fasta"

MaizeTSSv4_control_train = "dataset/MaizeTSSv4/train_control_maize.fasta"
MaizeTSSv4_control_val = "dataset/MaizeTSSv4/val_control_maize.fasta"
MaizeTSSv4_control_test = "dataset/MaizeTSSv4/test_control_maize.fasta"

control_seqs_arapdopsis_train = parse_control_fasta(ArabTSSv2_control_train)
control_seqs_arapdopsis_val = parse_control_fasta(ArabTSSv2_control_val)
control_seqs_arapdopsis_test = parse_control_fasta(ArabTSSv2_control_test)

control_seqs_maize_train = parse_control_fasta(MaizeTSSv4_control_train)
control_seqs_maize_val = parse_control_fasta(MaizeTSSv4_control_val)
control_seqs_maize_test = parse_control_fasta(MaizeTSSv4_control_test)

if args.dataset == 'arabidopsis':
    train_tss_dataset = TSSDataset(ArabTSSv2_train, "arabidopsis", is_training=True)
    val_tss_dataset = TSSDataset(ArabTSSv2_val, "arabidopsis", is_training=False)
    test_tss_dataset = TSSDataset(ArabTSSv2_test, "arabidopsis", is_training=False)

    # Print number of samples for TSSDataset
    print(f"Number of samples in train_tss_dataset: {len(train_tss_dataset)}")
    print(f"Number of samples in val_tss_dataset: {len(val_tss_dataset)}")
    print(f"Number of samples in test_tss_dataset: {len(test_tss_dataset)}")
    
    # Assuming max_len is an attribute of TSSDataset
    max_len = train_tss_dataset.max_len
    
    train_loader_tss = DataLoader(train_tss_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader_tss = DataLoader(val_tss_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader_tss = DataLoader(test_tss_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Pass max_len to ControlDataset
    train_control_dataset = ControlDataset(control_seqs_arapdopsis_train, max_len, is_training=True, target_len=len(train_tss_dataset), seed=SEED)
    val_control_dataset = ControlDataset(control_seqs_arapdopsis_train, max_len, is_training=True, target_len=len(train_tss_dataset), seed=SEED)
    #test_control_dataset = ControlDataset(control_seqs_arapdopsis_train, max_len, is_training=True, target_len=len(train_tss_dataset), seed=SEED)

    train_loader_control = DataLoader(train_control_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader_control = DataLoader(val_control_dataset, batch_size=64, shuffle=True, num_workers=4)
    #test_loader_control = DataLoader(test_control_dataset, batch_size=64, shuffle=True, num_workers=4)

if args.dataset == 'maize':
    # For maize
    train_tss_dataset = TSSDataset(MaizeTSSv4_train, "maize", is_training=True)
    val_tss_dataset = TSSDataset(MaizeTSSv4_val, "maize", is_training=False)
    test_tss_dataset = TSSDataset(MaizeTSSv4_test, "maize", is_training=False)
    
    # Print number of samples for TSSDataset
    print(f"Number of samples in maize train_tss_dataset: {len(train_tss_dataset)}")
    print(f"Number of samples in maize val_tss_dataset: {len(val_tss_dataset)}")
    print(f"Number of samples in maize test_tss_dataset: {len(test_tss_dataset)}")

    max_len = train_tss_dataset.max_len  # Assuming max_len is an attribute of TSSDataset

    # Create data loaders for maize
    train_loader_tss = DataLoader(train_tss_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader_tss = DataLoader(val_tss_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader_tss = DataLoader(test_tss_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    train_control_dataset = ControlDataset(control_seqs_maize_train, max_len, is_training=True, target_len=len(train_tss_dataset), seed=SEED)
    val_control_dataset = ControlDataset(control_seqs_maize_val, max_len, is_training=False, target_len=len(val_tss_dataset), seed=SEED)
    #test_control_dataset = ControlDataset(control_seqs_maize_test, max_len, is_training=False, target_len=len(test_tss_dataset), seed=SEED)

    train_loader_control = DataLoader(train_control_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader_control = DataLoader(val_control_dataset, batch_size=64, shuffle=True, num_workers=4)
    #test_loader_control = DataLoader(test_control_dataset, batch_size=64, shuffle=True, num_workers=4)

elif args.dataset == 'both':
    # Load Arabidopsis and Maize datasets separately for both training and validation sets
    train_tss_dataset_arab = TSSDataset(ArabTSSv2_train, "arabidopsis", is_training=True)
    train_tss_dataset_maize = TSSDataset(MaizeTSSv4_train, "maize", is_training=True)
    
    val_tss_dataset_arab = TSSDataset(ArabTSSv2_val, "arabidopsis", is_training=False)
    val_tss_dataset_maize = TSSDataset(MaizeTSSv4_val, "maize", is_training=False)
    
    # Combine the training and validation datasets
    train_tss_dataset = train_tss_dataset_arab + train_tss_dataset_maize
    val_tss_dataset = val_tss_dataset_arab + val_tss_dataset_maize
    
    # Calculate the max length across both datasets
    max_len = max(train_tss_dataset_arab.max_len, train_tss_dataset_maize.max_len)
    
    # Print number of samples for TSSDataset
    print(f"Number of samples in both train_tss_dataset: {len(train_tss_dataset)}")
    print(f"Number of samples in both val_tss_dataset: {len(val_tss_dataset)}")
    
    # Create data loaders for both training and validation sets
    train_loader_tss = DataLoader(train_tss_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader_tss = DataLoader(val_tss_dataset, batch_size=64, shuffle=True, num_workers=4)
    
    # For control dataset, combine Arabidopsis and Maize control sequences for both training and validation
    control_seqs_both_train = control_seqs_arapdopsis_train + control_seqs_maize_train
    control_seqs_both_val = control_seqs_arapdopsis_val + control_seqs_maize_val
    
    train_control_dataset = ControlDataset(control_seqs_both_train, max_len, is_training=True, target_len=len(train_tss_dataset), seed=SEED)
    val_control_dataset = ControlDataset(control_seqs_both_val, max_len, is_training=False, target_len=len(val_tss_dataset), seed=SEED)
    
    train_loader_control = DataLoader(train_control_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader_control = DataLoader(val_control_dataset, batch_size=64, shuffle=True, num_workers=4)


class CombinedModel(torch.nn.Module):
    def __init__(self, target_length=256, dropout=0.01):
        super().__init__()

        self.enformer  = Enformer.from_hparams(
            dim = 1536,
            depth = 11,
            heads = 8,
            output_heads = dict(prob = 256, tss = 1),
            target_length = 512,
        )
        self.gelu = nn.PReLU()
        self.classification_head = nn.Sequential(
            nn.Linear(3072, 128),
            self.gelu,
            nn.Dropout(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, one_hot_sequence):
        # Get the output and embeddings from the enformer
        output, embeddings = self.enformer(one_hot_sequence, return_embeddings=True)
        embeddings = embeddings.mean(dim=1)
        classification = output['prob'].mean(dim=1)
        classification_clamped = torch.clamp(classification, 0, 1)

        # Classifier Head
        classifier = self.classification_head(embeddings)

        out = classification_clamped

        return out,classifier.squeeze(-1)
    
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


def parse_sequences(fasta_file, seq_type):
    origin = {}
    sequences = {}

    with open(fasta_file, 'r') as f:
        seq_id = ''
        seq_data = ''
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq_id:
                    sequences[seq_id] = seq_data
                    origin[seq_id] = seq_type
                    
                seq_id = line[1:]
                seq_data = ''
            else:
                seq_data += line
        if seq_id:
            sequences[seq_id] = seq_data
            origin[seq_id] = seq_type

    tss_ids = list(sequences.keys())

    return sequences, origin, tss_ids

lr = 0.000001
wd = 0.0001
tolerance_values = list(range(1, 55))

combined_model = CombinedModel(device).to(device)
# Load the pre-trained model
#model_path = 'GOOD_MAIZE/models/maize_epoch_OK3.pth'
model_path = 'GOOD_ARAB/models/arabidopsis_BOM4.pth'

# Load the pre-trained model state dictionary
state_dict = torch.load(model_path, map_location=device)

# Load the filtered state dictionary, set strict=False to ignore missing keys
combined_model.load_state_dict(state_dict)

# Wrap the model with DataParallel for multi-GPU training
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    combined_model = nn.DataParallel(combined_model)

combined_model = combined_model.to(device)

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def poisson_loss(pred, target):
    return (pred - target * log(pred)).mean()

optimizer = torch.optim.AdamW(combined_model.parameters(), lr=lr, weight_decay=wd)

def log_cosh_loss(pred, target):
    return torch.mean(torch.log(torch.cosh(pred - target)))

def distribution_iou_loss(preds, labels, epsilon=1e-8):
    # Ensure positive values only, clamp with epsilon to avoid division by zero
    preds_clamped = torch.clamp(preds, min=epsilon)
    labels_clamped = torch.clamp(labels, min=epsilon)

    # Intersection: Element-wise minimum between preds and labels
    intersection = torch.minimum(preds_clamped, labels_clamped).sum(dim=1)
    
    # Union: Element-wise maximum between preds and labels, minus intersection to avoid double counting
    union = torch.maximum(preds_clamped, labels_clamped).sum(dim=1) - intersection
    
    # IoU Score: Intersection over Union
    iou_score = intersection / (union + epsilon)  # Add epsilon to avoid division by zero
    
    # IoU Loss: Subtract from 1 to minimize
    loss = 1 - iou_score
    
    # Return the mean loss
    return loss.mean()


def sequence_to_onehot(seq):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    return [mapping.get(nucleotide, [0, 0, 0, 0]) for nucleotide in seq] 
    
classification_criterion_yes_no = nn.BCELoss()
classification_criterion = poisson_loss

combined_model.eval()

def read_sequence_from_file(file_path):
    with open(file_path, 'r') as file:
        sequence = file.read().strip()
    return sequence

def process_sequence_with_sliding_window_and_plot(sequence, file_name, window_size=256, step_size=32, plot_dir="plots"):
    sequence_length = len(sequence)
    full_preds = np.zeros(sequence_length)  # Initialize with zeros to accumulate predictions

    # Ensure the output directory exists
    os.makedirs(plot_dir, exist_ok=True)

    # Preparing the batch
    windows_batch = []
    start_positions = []  # Keep track of start positions for each window

    for start in range(0, sequence_length - window_size + 1, step_size):
        end = start + window_size
        window = sequence[start:end]
        
        seq_one_hot = sequence_to_onehot(window)  # Assuming this function is defined elsewhere
        windows_batch.append(seq_one_hot)
        start_positions.append(start)

    # Convert list of windows to a tensor for batch processing
    batch_tensor = torch.tensor(windows_batch, dtype=torch.float32).view(-1, window_size, 4).to(device)

    with torch.no_grad():
        # Assuming your model can handle batches and outputs predictions in a compatible format
        tss_preds_batch, classifier_preds_batch = combined_model(batch_tensor)

    # Iterate through batch predictions
    for i, (tss_preds, classifier_preds) in enumerate(zip(tss_preds_batch, classifier_preds_batch)):
        scaled_preds_numpy = tss_preds.detach().cpu().numpy() * (classifier_preds.item() * 10)
        start = start_positions[i]
        full_preds[start:start+window_size] += scaled_preds_numpy

    max_pred_index = np.argmax(full_preds)
    ground_truth_pos = 750  # This seems to be your fixed reference point

    print(f"Ground Truth Position: {ground_truth_pos}, Max Prediction Index: {max_pred_index}")

    diff = abs(max_pred_index - ground_truth_pos)
    print(f"Difference: {diff}")
    
    '''# Plotting
    plt.figure(figsize=(20, 5))
    plt.plot(full_preds, color='blue')
    plt.axvline(x=max_pred_index, color='red', linestyle='--')  # Marking the max prediction index
    plt.axvline(x=ground_truth_pos, color='green', linestyle='--')  # Marking the max prediction index

    plt.title(f'Prediction Curve for {file_name}')
    plt.xlabel('Position in Sequence')
    plt.ylabel('Prediction Value')
    plt.savefig(os.path.join(plot_dir, f'plot_{file_name}.png'))
    plt.close()'''

    return full_preds, diff


# Set the path to the directory containing the text files
directory_path = 'experiment_arab/'

# List all txt files in the directory
file_paths = glob.glob(os.path.join(directory_path, '*.txt'))

# Initialize an empty list to store prediction results
prediction_results = []

# Loop through each file, read the sequence, and process it

diffs = []
for file_path in tqdm(file_paths, desc="Processing Files"):

    sequence = read_sequence_from_file(file_path)
    # Extract just the filename without the extension to use in the plot title and filename
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    predictions, diff = process_sequence_with_sliding_window_and_plot(sequence, file_name)
    diffs.append(diff)

specified_tolerances = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 400, 500]
comparison = pd.DataFrame(diffs, columns=["position_diff"])
overall_mean_accuracies = {}

total_predictions = len(comparison)
for tolerance in specified_tolerances:
    matches_within_tolerance = comparison[comparison["position_diff"] <= tolerance].shape[0]
    overall_mean_accuracies[tolerance] = matches_within_tolerance / total_predictions

accuracy_table = PrettyTable()
accuracy_table.field_names = ["Tolerance", "Mean Accuracy"]
for tolerance, mean_accuracy in overall_mean_accuracies.items():
    accuracy_table.add_row([tolerance, f"{mean_accuracy:.4f}"])
print(accuracy_table)

'''
sequences, origin, tss_ids = parse_sequences(MaizeTSSv4_test, "maize")

# Assuming combined_model, sequence_to_onehot, device, sequences, tss_ids are defined
window_size = 256
step_size = 16

# Variables to keep track of prediction statistics
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

for tss_id in tqdm(tss_ids, desc="Processing Sequences"):
    downstream, tss, upstream = sequences[tss_id].split('|')
    sequence = downstream + tss + upstream
    sequence_size = len(sequence)
    tss_center_position = len(downstream) + len(tss) // 2

    # Initialize a large array for predictions with zeros
    # and adjust its size to accommodate the combined length of downstream, tss, and upstream
    combined_sequence_length = len(downstream + tss + upstream)
    full_preds = np.zeros(combined_sequence_length)

    probs = []

    plt.figure(figsize=(20, 5))  # Adjusted figure size for better visualization
    for start in range(0, sequence_size - window_size + 1, step_size):
        end = start + window_size
        window = sequence[start:end]
        tss_in_window = start <= tss_center_position < end

        seq_one_hot = sequence_to_onehot(window)

        onehot_seq_tensor = torch.tensor(seq_one_hot, dtype=torch.float32).view(256, 4).to(device)
        onehot_seq_tensor_batched = onehot_seq_tensor.unsqueeze(0)

        tss_preds_tss, classifier_preds_tss = combined_model(onehot_seq_tensor_batched)

        # Convert the first predictions to numpy for plotting
        preds_numpy = tss_preds_tss[0].detach().cpu().numpy()

        probs.append(classifier_preds_tss.item())
        is_predicted_tss = classifier_preds_tss.item() > 0.9

        if(is_predicted_tss):
            # Insert the predictions into the full_preds array at the correct position
            full_preds[start:start+window_size] += preds_numpy
            #plt.plot(full_preds, color='blue')

        # Update statistics
        if tss_in_window and is_predicted_tss:
            true_positives += 1
        elif not tss_in_window and is_predicted_tss:
            false_positives += 1
        elif tss_in_window and not is_predicted_tss:
            false_negatives += 1
        elif not tss_in_window and not is_predicted_tss:
            true_negatives += 1

    plt.plot(full_preds, color='blue')
    plt.title('Prediction Curve for the Entire Sequence')
    plt.xlabel('Position in Sequence')
    plt.ylabel('Prediction Value')
    plt.legend()
    plt.savefig(f'w16/plot_tss_{tss_id}.png')  # Save each plot with a unique name
    plt.close()
    #input()
       

# Calculate additional metrics
total_predictions = true_positives + false_positives + true_negatives + false_negatives
accuracy = (true_positives + true_negatives) / total_predictions if total_predictions > 0 else 0
precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

# Print summary statistics
print("Summary of Predictions:")
print(f"True Positives: {true_positives}")
print(f"False Positives: {false_positives}")
print(f"True Negatives: {true_negatives}")
print(f"False Negatives: {false_negatives}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")

input()


val_loss = 0.0
accuracy_values = {
    'arabidopsis': {tolerance: [] for tolerance in tolerance_values},
    'maize': {tolerance: [] for tolerance in tolerance_values}
}
total_correct_classifier = 0
total_samples = 0
correct_classifier_arabidopsis = 0
samples_arabidopsis = 0
correct_classifier_maize = 0
samples_maize = 0
samples_control = 0
val_predictions = []
val_true_values = []
correct_predictions_arabidopsis = 0
correct_predictions_maize = 0
total_tss_sequences = 0
total_control_sequences = 0
classifier_outputs_tss = []
classifier_outputs_control = []
total_correct_predictions = 0
correct_predictions_control = 0

with torch.no_grad():
    val_loss = 0.0
    correct_predictions_count = 0  # Initialize counter for correct predictions
    incorrect_predictions_count = 0  # Initialize counter for incorrect predictions

    # Flags to track whether we've plotted examples for incorrect predictions
    plotted_for_gt_05 = False
    plotted_for_le_05 = False

    accuracy_list = []
    # Loop through validation data
    for i, ((sequences_tss, positions_tss, _, seq_types_tss, combined_seq_tss, prob_dist_tensor_tss), (sequences_control, positions_control, _, seq_types_control, seq_control, prob_dist_tensor_control)) in enumerate(zip(val_loader_tss, val_loader_control)):
        sequences = torch.cat([sequences_tss, sequences_control], dim=0).to(device)
        labels = torch.cat([prob_dist_tensor_tss, prob_dist_tensor_control], dim=0).to(device)
        
        labels_tss_yes_no = torch.ones(sequences_tss.size(0)).to(device)
        labels_control_yes_no = torch.zeros(sequences_control.size(0)).to(device)
        shuffled_labels_yes_no = torch.cat([labels_tss_yes_no, labels_control_yes_no], dim=0)

        preds, yes_no = combined_model(sequences)
        max_prob = torch.mean(preds, dim=1)

        max_pred_indices = torch.argmax(preds, dim=1)
        num_tss_sequences = sequences_tss.size(0)

        # Split max_pred_indices into two parts: one for TSS and one for control
        max_pred_indices_tss = max_pred_indices[:num_tss_sequences]
        max_pred_indices_control = max_pred_indices[num_tss_sequences:]
        
        # Calculate losses
        loss = classification_criterion(preds, labels)
        loss_yes_no = classification_criterion_yes_no(yes_no, shuffled_labels_yes_no)
        val_loss += loss.item() + loss_yes_no.item()

        # Now working with binary classification results (yes_no)
        combined_seq_types = seq_types_tss + seq_types_control
        combined_predictions = yes_no

        # Iterate over each prediction and its corresponding sequence type
        for idx, (prediction, seq_type) in enumerate(zip(combined_predictions, combined_seq_types)):
            is_tss = idx < len(seq_types_tss)  # First half are TSS sequences

            # Determine correctness
            is_correct = ((prediction > 0.5).item() if is_tss else (prediction <= 0.5).item())

            if not is_correct:
                incorrect_predictions_count += 1
                if not plotted_for_gt_05 and prediction > 0.5:
                    plot_example(preds[idx].cpu().numpy(), labels[idx].cpu().numpy(), 'incorrect_gt_05.png')
                    plotted_for_gt_05 = True
                    print("prediction > 0.5",max_prob[idx], yes_no[idx])
                elif not plotted_for_le_05 and prediction <= 0.5:
                    plot_example(preds[idx].cpu().numpy(), labels[idx].cpu().numpy(), 'incorrect_le_05.png')
                    plotted_for_le_05 = True
            else:
                correct_predictions_count += 1

        # Output the counts after the loop
        print(f"Correct predictions: {correct_predictions_count}")
        print(f"Incorrect predictions: {incorrect_predictions_count}")

        total_predictions = correct_predictions_count + incorrect_predictions_count

        accuracy = correct_predictions_count / total_predictions
        accuracy_list.append(accuracy)

        total_samples += labels.size(0)
        for tolerance in tolerance_values:
            diff = torch.abs(positions_tss.to(device) - max_pred_indices_tss[:positions_tss.size(0)])
            correct_predictions = torch.sum(diff <= tolerance).item()
            accuracy = correct_predictions / positions_tss.size(0)

            for individual_seq_type in seq_types_tss:
                accuracy_values[individual_seq_type][tolerance].append(accuracy)

        total_samples += labels.size(0)
        val_predictions.extend(max_pred_indices_tss[:max_pred_indices_tss.size(0)].cpu().detach().numpy())
        val_true_values.extend(positions_tss.cpu().detach().numpy())

    avg_val_loss = val_loss / total_samples
    # Calculate the average validation accuracy
    if len(accuracy_list) > 0:
        avg_val_acc = sum(accuracy_list) / len(accuracy_list)
    else:
        avg_val_acc = 0  # Handle case with no accuracy measurements (should generally not happen)

    print(f"Average Validation Accuracy: {avg_val_acc:.4f}")
    print(f"Validation Loss: {avg_val_loss:.8f}")

mean_accuracies_arabidopsis = [np.mean(accuracy_values['arabidopsis'][tolerance]) for tolerance in tolerance_values]
mean_accuracies_maize = [np.mean(accuracy_values['maize'][tolerance]) for tolerance in tolerance_values]

if args.dataset == 'both':
    overall_mean_accuracies = [(a + m) / 2 for a, m in zip(mean_accuracies_arabidopsis, mean_accuracies_maize)]
elif args.dataset == 'arabidopsis':
    overall_mean_accuracies = mean_accuracies_arabidopsis
elif args.dataset == 'maize':
    overall_mean_accuracies = mean_accuracies_maize

current_sum_accuracy = sum(overall_mean_accuracies)
val_classifier_accuracy = total_correct_classifier / total_samples

specified_tolerances = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]
accuracy_table = PrettyTable()
accuracy_table.field_names = ["Model", "Epoch", "Tolerance (nt)", "Overall TSS Accuracy"]

print(f"[Model {args.dataset}]: Current Sum Accuracy: {current_sum_accuracy:.4f}")

for tolerance in specified_tolerances:
    row = [
        args.dataset,
        0 + 1,
        tolerance,
        f"{overall_mean_accuracies[tolerance]:.4f}"
    ]
    accuracy_table.add_row(row)
print(accuracy_table)
'''
