import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.ticker import ScalarFormatter

def plot_sample_sequences(data_dir, classes, samples_per_class=2):
    plt.figure(figsize=(15, 8))
    subplot_idx = 1
    for class_idx, cls in enumerate(classes):
        class_files = sorted([f for f in os.listdir(data_dir) if cls in f and f.endswith('.csv')])
        np.random.shuffle(class_files)
        for i in range(min(samples_per_class, len(class_files))):
            df = pd.read_csv(os.path.join(data_dir, class_files[i]))
            sequence = df['sfc_encoded'].values
            plt.subplot(len(classes), samples_per_class, subplot_idx)
            plt.plot(sequence)
            plt.title(f"{cls}")
            plt.tight_layout()
            subplot_idx += 1
    plt.suptitle("Sample Sequences per Class", fontsize=16, y=1.02)
    plt.savefig("data/sequences_sample.png")

def plot_sample_all_columns(data_dir, classes, samples_per_class=2):
    sample_files = []

    # Pick one CSV file per class
    for cls in classes:
        class_files = sorted([f for f in os.listdir(data_dir) if cls in f and f.endswith('.csv')])
        np.random.shuffle(class_files)
        sample_files.append(os.path.join(data_dir, class_files[0]))

    if not sample_files:
        print("No samples to plot.")
        return

    # Read the first file to determine number of columns
    example_df = pd.read_csv(sample_files[0])
    data_columns = example_df.columns
    num_cols = len(data_columns)
    num_rows = len(sample_files)

    # Create subplot grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 3 * num_rows), squeeze=False)

    for row_idx, file_path in enumerate(sample_files):
        df = pd.read_csv(file_path)
        for col_idx, col_name in enumerate(df.columns):
            ax = axes[row_idx, col_idx]
            if col_name == 'sfc_encoded':
                # Scatter plot with reversed axes
                ax.scatter(df[col_name], np.arange(len(df)), s=4)
                if row_idx == 0:
                    ax.set_title(col_name)
                ax.set_ylabel("Time Step")

                ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            else:
                ax.plot(df[col_name], linewidth=1)
                if row_idx == 0:
                    ax.set_title(col_name)
                if col_idx == 0:
                    ax.set_ylabel(os.path.basename(file_path), fontsize=8)

            ax.tick_params(labelsize=8)

    plt.tight_layout()
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/sequences_all_columns.png")

def plot_sample_trajectories(data_dir, classes, samples_per_class=1):
    total_plots = len(classes) * samples_per_class
    fig, axes = plt.subplots(1, total_plots, figsize=(5 * total_plots, 5), squeeze=False)

    plot_index = 0
    for cls in classes:
        class_files = sorted([f for f in os.listdir(data_dir) if cls in f and f.endswith('.csv')])
        np.random.shuffle(class_files)

        for i in range(samples_per_class):
            ax = axes[0, plot_index]

            if i >= len(class_files):
                ax.axis('off')
                plot_index += 1
                continue

            df = pd.read_csv(os.path.join(data_dir, class_files[i]))

            x = df['positions_x']
            y = df['positions_y']
            speed = df['speed']

            # Prepare trajectory segments
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Create line collection with speed encoding
            lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(speed.min(), speed.max()), linewidth=3)
            lc.set_array(speed[:-1])
            ax.add_collection(lc)

            # Mark final point
            ax.plot(x.iloc[-1], y.iloc[-1], marker='o', color='black', markeredgecolor='yellow', markeredgewidth=1.5, markersize=10)

            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(y.min(), y.max())
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title(f"{cls}")
            ax.axis('equal')
            ax.grid(True)

            # Add individual colorbar
            cbar = fig.colorbar(lc, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
            cbar.set_label("Speed (km/h)")

            plot_index += 1

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("data/all_sample_trajectories.png")


if __name__ == "__main__":
    classes = ['lefturn', 'rightturn', 'lanechange', 'ra']
    plot_sample_sequences("data/data_csv", classes=classes)
    plot_sample_trajectories("data/data_csv", classes=classes)
    plot_sample_all_columns("data/data_csv", classes=classes)
