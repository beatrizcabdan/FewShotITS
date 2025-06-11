import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_sample_sequences(data_dir, classes=['normal', 'lefturn', 'rightturn', 'noisy'], samples_per_class=2):
    plt.figure(figsize=(15, 8))
    subplot_idx = 1
    for class_idx, cls in enumerate(classes):
        class_files = sorted([f for f in os.listdir(data_dir) if cls in f and f.endswith('.csv')])
        np.random.shuffle(class_files)
        for i in range(min(samples_per_class, len(class_files))):
            df = pd.read_csv(os.path.join(data_dir, class_files[i]))
            sequence = df['value'].values
            plt.subplot(len(classes), samples_per_class, subplot_idx)
            plt.plot(sequence)
            plt.title(f"{cls} - sample {i+1}")
            plt.tight_layout()
            subplot_idx += 1
    plt.suptitle("Sample Sequences per Class", fontsize=16, y=1.02)
    plt.savefig("data/sequences_sample.png")


if __name__ == "__main__":
    plot_sample_sequences("data/data_csv")
