def find_best_epoch(file_path):
    with open(file_path, 'r') as file:
        next(file)  # Skip the header
        best_epoch = None
        max_corr_coef = -float('inf')  # Start with the smallest possible float

        for line in file:
            parts = line.strip().split(',')
            epoch = int(parts[0])
            corr_coef = float(parts[2])

            if corr_coef > max_corr_coef:
                max_corr_coef = corr_coef
                best_epoch = epoch

    return best_epoch

# Usage
file_path = 'training_log.txt'
best_epoch = find_best_epoch(file_path)
print(f'The best epoch for Avg Corr Coef is: {best_epoch}')

