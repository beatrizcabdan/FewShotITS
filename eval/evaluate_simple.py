import torch.optim as optim
from model.encoder import TinyCNNEncoder
from train.train_fewshot import train_one_episode
import matplotlib.pyplot as plt
from utils.loader import load_episode_custom


def plot_accuracy_through_episodes(accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o', linestyle='-')
    plt.title("Classification accuracy through episodes")
    plt.xlabel("Episode")
    plt.ylabel("Accuracy")
    plt.xticks(range(1, len(accuracies) + 1))
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("eval/accuracy_plot.png")


if __name__ == "__main__":
    encoder = TinyCNNEncoder()
    optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    episodes = 20
    included_classes = ['normal', 'lefturn', 'rightturn', 'noisy']

    accuracies = []
    # full training runner with synthetic episodes (to improve embeddings)
    for episode in range(episodes):
        data = load_episode_custom(data_dir="data/data_csv", included_classes=included_classes, shot=10, query_per_class=5)  # randomly sampled support and query sets
        loss, acc = train_one_episode(encoder, data, optimizer)  # updates the encoder’s weights using gradient descent
        print(f"Episode {episode + 1}: Loss = {loss:.4f}, Accuracy = {acc:.2%}")
        accuracies.append(acc)

    plot_accuracy_through_episodes(accuracies)