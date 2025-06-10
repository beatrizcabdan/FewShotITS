import numpy as np
import os
import pandas as pd


def generate_sequence(class_type, noise_std, length=30):
    base = np.ones(length) * 50
    if class_type == 'normal':
        sequence = base
    elif class_type == 'spike':
        base[length // 2] += 15 + np.random.normal(0, 3)
        sequence = base
    elif class_type == 'drop':
        base[length // 2] -= 15 + np.random.normal(0, 3)
        sequence = base
    elif class_type == 'pattern':
        pattern = np.tile([5, -5], length // 2)
        sequence = base + pattern
    else:
        sequence = base

    # Add noise
    noise = np.random.normal(0, noise_std, size=length)
    # noise = np.zeros(length) * 50
    return sequence + noise


def save_sequences(output_dir="data/data_csv", num_samples=20, noise_std=2.0):
    os.makedirs(output_dir, exist_ok=True)
    classes = ['normal', 'spike', 'drop', 'pattern']
    for cls in classes:
        for i in range(num_samples):
            seq = generate_sequence(cls, noise_std=noise_std)
            df = pd.DataFrame(seq, columns=["value"])
            df.to_csv(os.path.join(output_dir, f"{cls}_{i}.csv"), index=False)


if __name__ == "__main__":
    save_sequences(noise_std=3)  # synthetic sequence generator