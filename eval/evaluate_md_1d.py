import torch.optim as optim
from model.encoder import TinyMLEncoder, TinyCNNPlusEncoder
from train.train_fewshot import train_one_episode
import matplotlib.pyplot as plt
from train.episodic_loader import load_episode_1d, load_episode_md
import numpy as np


def plot_accuracy_through_episodes(accuracies_1d, accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(accuracies_1d) + 1), accuracies_1d, marker='o', linestyle='-', label="SFC-encoded data")
    plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o', linestyle='-', label="Original multimodal data")
    plt.title("Classification accuracy through episodes")
    plt.xlabel("Episode")
    plt.ylabel("Accuracy")
    plt.xticks(range(1, len(accuracies) + 1))
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("evaluate_md_1d.png")


if __name__ == "__main__":
    episodes = 20
    included_classes = ['lefturn', 'rightturn', 'ra', 'lanechange']
    query_per_class = 100
    shot = 5
    lr = 0.01

    # MD RESULTS
    encoder = TinyCNNPlusEncoder(input_channels=6, output_dim=4)
    optimizer = optim.Adam(encoder.parameters(), lr=lr)

    accuracies = []
    for episode in range(episodes):
        data = load_episode_md(data_dir="../data/data_csv", included_classes=included_classes, shot=shot, query_per_class=query_per_class)
        loss, acc = train_one_episode(encoder, data, optimizer)
        accuracies.append(acc)

    mean_acc = np.mean(accuracies[-10:])
    std_acc = np.std(accuracies[-10:])
    print(f"Evaluation MD = {mean_acc:.2%} +- {std_acc:.2%}")
    continu = accuracies[0] > .95

    # 1D TINYML RESULTS
    encoder = TinyMLEncoder(output_dim=4, out_channels=4)
    optimizer = optim.Adam(encoder.parameters(), lr=lr)

    accuracies_1d = []
    for episode in range(episodes):
        data = load_episode_1d(data_dir="../data/data_csv", included_classes=included_classes, shot=shot, query_per_class=query_per_class)  # randomly sampled support and query sets
        loss, acc = train_one_episode(encoder, data, optimizer)
        accuracies_1d.append(acc)

    mean_acc = np.mean(accuracies_1d[-10:])
    std_acc = np.std(accuracies_1d[-10:])
    print(f"Evaluation 1Dt = {mean_acc:.2%} +- {std_acc:.2%}")

    plot_accuracy_through_episodes(accuracies_1d, accuracies)




