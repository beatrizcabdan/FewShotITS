import torch
import torch.optim as optim
import numpy as np
from model.encoder import TinyCNNEncoder
from train.train_fewshot import train_one_episode
from model.proto_net import compute_prototypes, classify_queries
import matplotlib.pyplot as plt
from utils.loader import load_episode_custom


def load_data_by_class(data_dir, exclude_class=None):
    classes = ['normal', 'spike', 'drop', 'pattern']
    if exclude_class:
        classes = [cls for cls in classes if cls != exclude_class]
    return classes


def evaluate_on_held_out_class(data_dir, held_out, episodes=10, eval_repeats=10):
    included_classes = load_data_by_class(data_dir, exclude_class=held_out)
    encoder = TinyCNNEncoder()
    optimizer = optim.Adam(encoder.parameters(), lr=0.001)

    # print(f"Training on classes: {included_classes}, holding out: {held_out}")
    for episode in range(episodes):
        data = load_episode_custom(data_dir, included_classes)
        train_one_episode(encoder, data, optimizer)

    # print(f"Evaluating adaptation to held-out class: {held_out}")
    distractor = np.random.choice([c for c in ['normal', 'spike', 'drop', 'pattern'] if c != held_out])
    held_out_classes = [held_out, distractor]
    adaptation_accuracies = []

    for _ in range(eval_repeats):
        support_x, support_y, query_x, query_y = load_episode_custom(data_dir, held_out_classes)
        with torch.no_grad():
            support_embed = encoder(support_x)
            query_embed = encoder(query_x)
            prototypes, classes = compute_prototypes(support_embed, support_y)
            dists = classify_queries(query_embed, prototypes)
            preds = torch.argmin(dists, dim=1)

            # accuracy is based on actual label matching
            classes = torch.unique(support_y)
            true = torch.tensor([classes.tolist().index(label.item()) for label in query_y])
            acc = (preds == true).float().mean().item()
            adaptation_accuracies.append(acc)

    mean_acc = np.mean(adaptation_accuracies)
    std_acc = np.std(adaptation_accuracies)
    print(f"Held-out class: {held_out}, Accuracy: {mean_acc*100:.2f}%, Std: {std_acc*100:.2f}%")
    return mean_acc, std_acc


def evaluate_all_held_out(data_dir="data_csv", episodes=10, eval_repeats=10):
    classes = ['normal', 'spike', 'drop', 'pattern']
    results = {}
    errors = {}

    for held_out in classes:
        mean_acc, std_acc = evaluate_on_held_out_class(data_dir, held_out, episodes, eval_repeats)
        results[held_out] = mean_acc
        errors[held_out] = std_acc

    # Plot with error bars
    plt.figure(figsize=(8, 5))
    plt.bar(results.keys(), results.values(), yerr=errors.values(), capsize=8)
    plt.ylim(0, 1.1)
    plt.ylabel("Accuracy")
    plt.title("Accuracy per held-out class")
    plt.savefig("held_out_adaptation_accuracy.png")


if __name__ == "__main__":
    evaluate_all_held_out(data_dir="../data/data_csv", eval_repeats=10)