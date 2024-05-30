import pandas as pd
from Bio import SeqIO
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
from Bio import SeqIO
import re
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.lines as mlines
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from sklearn.cluster import KMeans
from collections import Counter
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import warnings
import matplotlib.colors as mcolors
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import confusion_matrix, roc_curve, auc
import itertools
import matplotlib
import math
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib import rc

matplotlib.use('Agg')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


'''def plot_accuracy_256(tolerance_values, overall_accuracies, mae, mse, classifier_accuracy, img_file):
    plt.figure(figsize=(10, 8))
    classifier_accuracy *= 100 
    # Convert accuracies to percentage
    overall_accuracies = [accuracy * 100 for accuracy in overall_accuracies]
    accuracy_subset = [accuracy for t, accuracy in zip(tolerance_values, overall_accuracies) if 1 <= t <= 100]

    # Create a colormap that goes from red (bad accuracy) to green (good accuracy)
    cmap = mcolors.LinearSegmentedColormap.from_list("redGreen", ["red", "green"])
    norm = mcolors.Normalize(vmin=0, vmax=100)  # Adjusted for percentage
    
    # Generate a matrix representation for color filling
    y_values = np.linspace(0, 100, len(tolerance_values))  # Adjusted for percentage
    accuracy_matrix = np.tile(overall_accuracies, (len(tolerance_values), 1))
    
    # Mask the values above the original accuracy points
    for i, y in enumerate(y_values):
        mask = y > overall_accuracies
        accuracy_matrix[i, mask] = np.nan  # Using NaN to mask the values
    
    # Display the masked matrix as an image
    plt.imshow(accuracy_matrix, aspect='auto', cmap=cmap, origin='lower', 
               extent=[tolerance_values[0], tolerance_values[-1], 0, 100], alpha=0.1)  # Adjusted for percentage
    
    # Plot the original accuracy points
    plt.plot(tolerance_values, overall_accuracies, label='Accuracy', color='red', linestyle='-')
    
    # Fit a 1st-degree polynomial and plot it (Ensure it doesn't exceed 100%)
    z = np.polyfit(tolerance_values, overall_accuracies, 1)
    p = np.poly1d(z)
    trend_values = p(tolerance_values)
    trend_values = np.clip(trend_values, 0, 100)
    plt.plot(tolerance_values, trend_values, "b--", label='Trendline')
    
    # Calculate the slope angle in degrees
    slope_angle = math.atan(z[0]) * (180/math.pi)
    
    # Annotate the slope angle in the middle of the trendline
    midpoint_index = len(tolerance_values) // 2
    plt.annotate(f"Slope: {slope_angle:.2f}°", (tolerance_values[midpoint_index], trend_values[midpoint_index]), textcoords="offset points", xytext=(5,-50), ha='center', fontsize=12, color='blue')
    
    # Annotate the graph with accuracy numbers at intervals of 20
    for i, (tolerance, accuracy) in enumerate(zip(tolerance_values, overall_accuracies)):
        if accuracy < 95:  # Adjusted for percentage
            if i % 15 == 0:
                plt.annotate(f"{accuracy:.2f}%", (tolerance, accuracy), textcoords="offset points", xytext=(-2,15), ha='center', fontsize=12)
        
    # Formatting
    min_accuracy = min(overall_accuracies)
    buffer = 5  # 5% buffer to ensure the minimum value isn't too close to the edge of the plot
    plt.ylim(min_accuracy - buffer, 100)
    
    plt.title('Accuracy Analysis at Different Tolerance Levels [1-256]', fontsize=18, fontweight='bold')
    plt.xlabel('Tolerance Level (nt)', fontsize=16)
    plt.ylabel('Accuracy (%)', fontsize=16)  # Adjusted for percentage
    plt.xticks(tolerance_values[::20], fontsize=14)  
    plt.yticks(fontsize=14)
    plt.legend(loc='lower right', fontsize=14, frameon=True, edgecolor='black')
    plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)

    # Add a text box with the metrics
    total_accuracy_subset = sum(accuracy_subset)/len(accuracy_subset) if len(accuracy_subset) > 0 else 0
    textstr = f'Sum of Position Accuracy [1-100]: {total_accuracy_subset:.2f}%\nClassifier: {classifier_accuracy:.2f}%\nMAE: {mae:.2f}\nMSE: {mse:.2f}'

    #textstr = f'Total Accuracy (positions): {sum(overall_accuracies)/len(overall_accuracies):.2f}%\nClassifier: {classifier_accuracy:.2f}%\nMAE: {mae:.2f}\nMSE: {mse:.2f}'

    props = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.7)
    #plt.text(0.693, 0.13, textstr, transform=plt.gca().transAxes, fontsize=14, verticalalignment='baseline', bbox=props, fontweight='bold')
    plt.text(0.548, 0.13, textstr, transform=plt.gca().transAxes, fontsize=14, verticalalignment='baseline', bbox=props, fontweight='bold')
    plt.tight_layout()
    #plt.savefig('accuracy.pdf', dpi=600, bbox_inches='tight')
    plt.savefig('imgs/accuracy_'+img_file+'.png', dpi=600)'''



def plot_accuracy(tolerance_values, overall_accuracies, mae, mse, classifier_accuracy, img_file):
    print(img_file)
    plt.figure(figsize=(9, 5))
    classifier_accuracy *= 100

    # Filter and convert data
    filtered_tolerance_values = [t for t in tolerance_values if 1 <= t <= 100]
    filtered_overall_accuracies = [accuracy * 100 for t, accuracy in zip(tolerance_values, overall_accuracies) if 1 <= t <= 100]
    
    # Calculate the mean accuracy
    mean_accuracy = sum(filtered_overall_accuracies) / len(filtered_overall_accuracies) if len(filtered_overall_accuracies) > 0 else 0
    
    # Generate custom x-ticks
    custom_ticks = list(range(1, 10, 3)) + list(range(20, 101, 10))
    
    # Plot the original accuracy points with a new color
    plt.plot(filtered_tolerance_values, filtered_overall_accuracies, label=f'Accuracy (Mean: {mean_accuracy:.2f}\%)', color='red', linestyle='-')
    
    # Fit a 1st-degree polynomial and plot it
    z = np.polyfit(filtered_tolerance_values, filtered_overall_accuracies, 1)
    p = np.poly1d(z)
    trend_values = p(filtered_tolerance_values)
    trend_values = np.clip(trend_values, 0, 100)
    plt.plot(filtered_tolerance_values, trend_values, "b--")#, label='Trendline')
    
    # Calculate the slope angle in degrees
    slope_angle = math.atan(z[0]) * (180/math.pi)
    midpoint_index = len(filtered_tolerance_values) // 2
    plt.annotate(f"Slope trendline: {slope_angle:.2f}°", (filtered_tolerance_values[midpoint_index], trend_values[midpoint_index]), textcoords="offset points", xytext=(5,-30), ha='center', fontsize=12, color='blue')

    # Annotate with dynamic position adjustment for custom_ticks
    for tolerance, accuracy in zip(filtered_tolerance_values, filtered_overall_accuracies):
        if tolerance in custom_ticks:
            offset = (0, 10)  
            if accuracy > 96:
                offset = (0, -17)  
            elif accuracy < 80:
                offset = (27, 0)
            plt.annotate(f"{accuracy:.2f}%", (tolerance, accuracy), textcoords="offset points", xytext=offset, ha='center', fontsize=12)
            plt.scatter(tolerance, accuracy, color='red', s=30)  # Circle marker
    
    # Formatting
    min_accuracy = min(filtered_overall_accuracies)
    buffer = 5
    plt.ylim(min_accuracy - buffer, 100)
    plt.title('Accuracy analysis at different tolerance levels [1-100] nt', fontsize=17, fontweight='bold')
    plt.xlabel('Tolerance Level (nt)', fontsize=16)
    plt.ylabel('Accuracy (\%)', fontsize=16)
    plt.xticks(custom_ticks, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='lower right', fontsize=14, frameon=True, edgecolor='black')
    plt.grid(axis='both', linestyle='--', linewidth=0.7, alpha=0.7)

    # Print MAE and MSE
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"Classifier: {classifier_accuracy:.2f}")
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('imgs/Accuracy '+img_file+'.png', bbox_inches='tight', dpi=600)
    plt.savefig('imgs/Accuracy '+img_file+'.pdf', bbox_inches='tight', pad_inches=0)


def plot_residuals(val_predictions, val_true_values, mae, mse, img_file):
    import matplotlib.colors as mcolors
    
    # Exclude non-TSS values from residuals plot
    mask = val_true_values != 0
    val_true_values = val_true_values[mask]
    val_predictions = val_predictions[mask]
    
    residuals = val_true_values - val_predictions
    abs_residuals = np.abs(residuals)
    cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", ['blue', 'red'])
    norm = plt.Normalize(vmin=min(abs_residuals), vmax=max(abs_residuals))

    plt.figure(figsize=(10, 6))
    plt.scatter(val_predictions, residuals, c=cmap(norm(abs_residuals)), alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Predicted')
    plt.ylabel('Residual')
    plt.title('Residuals vs Predicted')
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), label='Absolute Residual')
    textstr = f'MAE: {mae:.2f}\nMSE: {mse:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.80, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=props)
    plt.savefig('imgs/residuals_'+img_file+'.png')


def plot_bland_altman(val_predictions, val_true_values, img_file):
    # Exclude non-TSS values from Bland-Altman plot
    mask = val_true_values != 0
    val_true_values = val_true_values[mask]
    val_predictions = val_predictions[mask]

    mean = np.mean([val_true_values, val_predictions], axis=0)
    diff = val_true_values - val_predictions
    md = np.mean(diff)
    sd = np.std(diff, axis=0)
    abs_diff = np.abs(diff)
    norm = plt.Normalize(vmin=min(abs_diff), vmax=max(abs_diff))
    cmap = cm.get_cmap('coolwarm')

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(mean, diff, c=abs_diff, alpha=0.5, cmap=cmap, norm=norm)
    plt.axhline(md, color='gray', linestyle='--')
    plt.axhline(md + 1.96 * sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96 * sd, color='gray', linestyle='--')
    plt.title('Bland-Altman Plot')
    plt.xlabel('Mean Score (True and Pred)')
    plt.ylabel('Diff Score (True - Pred)')
    plt.colorbar(sc, label='Absolute Difference')
    plt.savefig('imgs/bland_altman_plot_'+img_file+'.png')


def scatter_plot(val_true_values, val_predictions, img_file):
    # Mask to exclude non-TSS sequences
    mask = val_true_values != -1
    
    # Data for TSS sequences
    val_true_tss = val_true_values[mask]
    val_predictions_tss = val_predictions[mask]
    
    distances = np.abs(val_true_tss - val_predictions_tss)
    cmap = cm.get_cmap('seismic')
    norm = plt.Normalize(vmin=min(distances), vmax=max(distances))

    plt.figure(figsize=(6, 5))
    plt.scatter(val_true_tss, val_predictions_tss, c=distances, cmap=cmap, norm=norm, alpha=0.5)
    plt.xlabel('True Center Positions')
    plt.ylabel('Predicted Center Positions')
    plt.title('Validation - Center Positions')
    plt.plot([min(val_true_tss), max(val_true_tss)], [min(val_true_tss), max(val_true_tss)], 'r')
    plt.colorbar(label='Distance from y=x')
    plt.tight_layout()
    plt.savefig('imgs/scatter_plot_'+img_file+'.png')
    


def plot_error_histogram(errors, img_file):
    # Calculate the absolute errors
    abs_errors = np.abs(errors)

    # Create a color map that goes from blue to red
    cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", ['blue', 'red'])

    # Normalize the absolute errors to range between 0 and 1 for the color map
    norm = plt.Normalize(vmin=min(abs_errors), vmax=max(abs_errors))

    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(errors, bins=30, alpha=0.5)

    # Now, we'll loop over our data points and color code each histogram bar
    for i in range(len(patches)):
        # Get the left and right side of the bin
        left, right = bins[i], bins[i+1]

        # Get the absolute value of the closest side to 0
        closest = min(abs(left), abs(right))

        # Color the bar based on its closest side to 0
        patches[i].set_facecolor(cmap(norm(closest)))

    plt.title('Histogram of Prediction Errors')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), label='Absolute Error')
    plt.grid()
    plt.savefig('imgs/hist_errors_'+img_file+'.png')
    

#NEW CODE
def plot_dna_window(dna_seq, true_tss_center, pred_tss_center, epoch, sequence_window=50):
    # Use a modern style
    #plt.style.use("seaborn-v0_8-whitegrid")


    base_colors = ['darkorange', 'steelblue', 'seagreen', 'firebrick'] 
    base_names = ['A', 'T', 'G', 'C']

    # Convert one-hot encoded sequence to color indices
    dna_seq_flat = np.argmax(dna_seq, axis=1)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 3))

    # Define the window around the true TSS to be visualized
    window_start = max(0, true_tss_center - sequence_window)
    window_end = min(len(dna_seq_flat), true_tss_center + sequence_window)

    # Calculate the window size
    window_size = window_end - window_start

    # Reduce the values of pred_tss_center and true_tss_center by window_start to match the windowed sequence
    pred_tss_center = pred_tss_center[0] - window_start if isinstance(pred_tss_center, np.ndarray) else pred_tss_center - window_start
    true_tss_center = true_tss_center - window_start

    # Plot the DNA sequence
    for i in range(window_start, window_end):
        base = dna_seq_flat[i]
        ax.add_patch(plt.Rectangle((i - window_start, 0), 1, 1, color=base_colors[base], alpha=0.5))
        ax.text(i - window_start + 0.5, 0.5, base_names[base], ha='center', va='center', fontsize=8)

    # Create custom legend
    custom_patches = [mpatches.Patch(color=base_colors[i], label=base_names[i]) for i in range(4)]
    legend = ax.legend(handles=custom_patches, loc='lower right')
    legend.get_frame().set_alpha(None)

    # Initialize warning flag
    no_warnings = True

    # Plot the predicted TSS center
    if 0 <= pred_tss_center < window_size:
        dx = min(2, window_size - pred_tss_center - 1)
        if dx > 0:
            line = mlines.Line2D([pred_tss_center, pred_tss_center], [0, 2], color='red')
            ax.add_line(line)
            ax.arrow(pred_tss_center, 2, dx, 0, head_width=0.15, head_length=1, fc='r', ec='r')
            ax.text(pred_tss_center + dx + 1.1, 1.9, 'Predicted TSS', fontsize=9, ha='left')
        else:
            print("Warning: Predicted TSS arrow is outside the plot.")
            no_warnings = False
    else:
        print("Warning: Predicted TSS is outside the plot.")
        no_warnings = False

    # Plot the true TSS center
    if 0 <= true_tss_center < window_size:
        dx = min(2, window_size - true_tss_center - 1)
        if dx > 0:
            line = mlines.Line2D([true_tss_center, true_tss_center], [0, -2], color='red')
            ax.add_line(line)
            ax.arrow(true_tss_center, -2, -dx, 0, head_width=0.15, head_length=1, fc='r', ec='r')
            ax.text(true_tss_center - dx - 1, -2.1, 'True TSS', fontsize=9, ha='right')
        else:
            print("Warning: True TSS arrow is outside the plot.")
            no_warnings = False
    else:
        print("Warning: True TSS is outside the plot.")
        no_warnings = False

    # Customize plot
    ax.set_yticks([])

    # Set the x-axis ticks and labels based on windowed sequence
    ax.set_xticks(np.arange(0, window_size, max(window_size // 10, 1)))
    ax.set_xticklabels(np.arange(window_start, window_end, max(window_size // 10, 1)))

    # Set the x-axis and y-axis limits
    ax.set_xlim(0, window_size)
    ax.set_ylim(-3, 3)

    # Add nucleotide label
    ax.set_xlabel("Nucleotides")

    # Add box around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)

    # Display the plot
    plt.title("DNA Sequence Window Surrounding the TSS", fontsize=16, pad=20)

    # Only save the figure if there were no warnings
    if no_warnings:
        plt.savefig('plot_dna/dna'+str(epoch)+".png", bbox_inches='tight')


def visualization_during_validation(sequences, positions, outputs, epoch):
    #clear_output(wait=True)
    # Select a random sequence from the batch
    sample_index = random.randint(0, sequences.shape[0] - 1)
    # Get the sequence and its true and predicted TSS center
    seq = sequences[sample_index].cpu().numpy()
    true_tss_center = positions[sample_index].cpu().numpy()
    pred_tss_center = outputs[sample_index].cpu().numpy()

    # Call the visualization function
    plot_dna_window(seq, true_tss_center, pred_tss_center,epoch, sequence_window=30 )



def plot_confusion_matrix(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels > 0.5)  # Threshold at 0.5
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = [0, 1]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')

def plot_kmeans(tss_lengths, title):
    lengths = np.array(tss_lengths).reshape(-1, 1)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(lengths)
    labels = kmeans.labels_

    # Get counts of each cluster
    counter = Counter(labels)
    
    # Create a color map for the three clusters
    colors = ['green', 'blue', 'red']
    label_colors = [colors[label] for label in labels]

    # Plot the histogram with colors representing clusters
    plt.hist([lengths[labels == i].flatten() for i in range(3)], bins=30, stacked=True, color=colors, alpha=0.7, label=[f'Cluster {i+1} - Count: {counter[i]}' for i in range(3)])

    # Annotate the cluster centers
    line_colors = ['darkgreen', 'darkblue', 'darkred']
    for i, center in enumerate(kmeans.cluster_centers_):
        plt.axvline(center[0], color=line_colors[i], linestyle='dashed', linewidth=1)
        text_offset = 5  # you can adjust this offset value as needed
        if center[0] < np.median(lengths):
            ha = 'right'
            offset = -text_offset
        else:
            ha = 'left'
            offset = text_offset
        plt.text(center[0] + offset, plt.gca().get_ylim()[1] * 0.5, f'Center {i+1}: {center[0]:.2f}', rotation=90, va='center', ha=ha, color='black')

    plt.title(title)
    plt.xlabel('TSS Length')
    plt.ylabel('Frequency')
    plt.xlim([0, 360])  # Limit the x-axis to 400
    plt.legend()
    plt.show()

def plot_roc_curve(true_labels, predicted_probs):
    fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')


# Convert lengths and labels to a DataFrame
def lengths_to_df(lengths, labels):
    return pd.DataFrame({'TSS Length': lengths, 'Cluster': labels})

# Get the lengths of the TSS sequences
def get_tss_lengths(tss_seqs):
    return [len(tss) for tss in tss_seqs.values()]

# Perform k-means clustering
def perform_kmeans(lengths):
    lengths_array = np.array(lengths).reshape(-1, 1)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(lengths_array)
    # Add 1 to the labels to make them 1-indexed
    return kmeans.labels_ + 1

# Box plot
# Create a color map for the three clusters
colors = ['green', 'blue', 'red']

# Box plot
def plot_boxplot(df, title):
    plt.figure(figsize=(10, 5))
    box_plot = sns.boxplot(x='Cluster', y='TSS Length', data=df, palette=colors)
    box_plot.set_xticklabels(['Cluster 1', 'Cluster 2', 'Cluster 3'])
    plt.title(title)
    
    # Get counts, mean, median, std, and IQR of each cluster
    counter = Counter(df['Cluster'])
    means = df.groupby('Cluster')['TSS Length'].mean()
    medians = df.groupby('Cluster')['TSS Length'].median()
    stds = df.groupby('Cluster')['TSS Length'].std()
    iqrs = df.groupby('Cluster')['TSS Length'].quantile(0.75) - df.groupby('Cluster')['TSS Length'].quantile(0.25)
    
    # Manually add legend with statistics
    handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
    plt.legend(handles, [
        f'Cluster {i+1} - Count: {counter[i+1]}, Mean: {means[i+1]:.2f}, Median: {medians[i+1]:.2f}, Std: {stds[i+1]:.2f}, IQR: {iqrs[i+1]:.2f}'
        for i in range(3)
    ])
    
    plt.show()

# Violin plot
def plot_violinplot(df, title):
    plt.figure(figsize=(10, 5))
    violin_plot = sns.violinplot(x='Cluster', y='TSS Length', data=df, palette=colors)
    violin_plot.set_xticklabels(['Cluster 1', 'Cluster 2', 'Cluster 3'])
    plt.title(title)
    
    # Get counts, mean, median, std, and IQR of each cluster
    counter = Counter(df['Cluster'])
    means = df.groupby('Cluster')['TSS Length'].mean()
    medians = df.groupby('Cluster')['TSS Length'].median()
    stds = df.groupby('Cluster')['TSS Length'].std()
    iqrs = df.groupby('Cluster')['TSS Length'].quantile(0.75) - df.groupby('Cluster')['TSS Length'].quantile(0.25)
    
    # Manually add legend with statistics
    handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
    plt.legend(handles, [
        f'Cluster {i+1} - Count: {counter[i+1]}, Mean: {means[i+1]:.2f}, Median: {medians[i+1]:.2f}, Std: {stds[i+1]:.2f}, IQR: {iqrs[i+1]:.2f}'
        for i in range(3)
    ])
    
    plt.show()


#train_lengths = get_tss_lengths(train_dataset.dataset.tss_seqs)
#train_labels = perform_kmeans(train_lengths)
#train_df = lengths_to_df(train_lengths, train_labels)