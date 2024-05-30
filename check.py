import torch
import os

def check_data_sizes(folder_path, fold_count=5):
    for fold in range(fold_count):
        train_data_path = os.path.join(folder_path, f'data/fold_{fold}', 'train_data.pt')
        validation_data_path = os.path.join(folder_path, f'data/fold_{fold}', 'validation_data.pt')

        # Load train data
        if os.path.exists(train_data_path):
            train_data = torch.load(train_data_path)
            print(f"Fold {fold} Train Data: {len(train_data.keys())} windows")
            first_key = next(iter(train_data))
            print(f"First data sample in Fold {fold} Train Data: Key '{first_key}', Sequence Length: {train_data[first_key]['sequence'].size()}, Distribution Length: {train_data[first_key]['distribution'].size()}")

        else:
            print(f"Fold {fold} Train Data: File not found")

        # Load validation data
        if os.path.exists(validation_data_path):
            validation_data = torch.load(validation_data_path)
            print(f"Fold {fold} Validation Data: {len(validation_data.keys())} windows")
        else:
            print(f"Fold {fold} Validation Data: File not found")

# Example usage
folder_path = 'athaliana/'  # Adjust this path to your dataset location
check_data_sizes(folder_path)

