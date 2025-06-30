import torch
import torch.optim as optim
import numpy as np
from model.encoder import TinyMLEncoder
from train.train_fewshot import train_one_episode
from model.proto_net import compute_prototypes, classify_queries
import matplotlib.pyplot as plt
from train.episodic_loader import load_episode_1d
import time
import pandas as pd

def load_data_by_class(classes, data_dir, exclude_class=None):
    if exclude_class:
        classes = [cls for cls in classes if cls != exclude_class]
    return classes


def evaluate_on_held_out_class(classes, data_dir, held_out, episodes=10, eval_repeats=10):
    included_classes = load_data_by_class(classes, data_dir, exclude_class=held_out)
    encoder = TinyMLEncoder()
    optimizer = optim.Adam(encoder.parameters(), lr=0.01)

    for episode in range(episodes):
        data = load_episode_1d(data_dir, included_classes, shot=5, query_per_class=50)
        train_one_episode(encoder, data, optimizer)

    distractor = np.random.choice([c for c in classes if c != held_out])
    held_out_classes = [held_out, distractor]
    adaptation_accuracies = []

    for _ in range(eval_repeats):
        support_x, support_y, query_x, query_y = load_episode_1d(data_dir, held_out_classes, shot=10, query_per_class=100)
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


def evaluate_all_held_out(classes, data_dir="data_csv", episodes=20, eval_repeats=10):
    results = {}
    errors = {}

    for held_out in classes:
        mean_acc, std_acc = evaluate_on_held_out_class(classes, data_dir, held_out, episodes, eval_repeats)
        results[held_out] = mean_acc
        errors[held_out] = std_acc

    # Plot with error bars
    plt.figure(figsize=(12, 5))
    plt.bar(results.keys(), results.values(), yerr=errors.values(), capsize=8)
    plt.ylim(0, 1.1)
    plt.ylabel("Accuracy")
    plt.title("Accuracy per held-out class")
    plt.savefig("evaluate_held_out.png")


def evaluate_acc_time(data_dir="../data/data_csv", episodes=50):
    # classes = ['lefturn', 'rightturn', 'ra', 'lanechange']
    # classes = ["lefttoright", "righttoleft", "stopinmiddle", "waittocross"]
    allclasses = ['lefturn', 'rightturn', 'ra', 'lanechange', "lefttoright", "righttoleft", "stopinmiddle", "waittocross"]
    way = "_8"

    dflist = []
    for held_out in allclasses*5:  # evaluate_on_held_out_class
        included_classes = load_data_by_class(allclasses, data_dir, exclude_class=held_out)

        encoder = TinyMLEncoder()
        optimizer = optim.Adam(encoder.parameters(), lr=0.01)

        # TRAINING
        for episode in range(episodes):
            data = load_episode_1d(data_dir, included_classes, shot=5, query_per_class=20)
            train_one_episode(encoder, data, optimizer)

        maxacc = 0
        for numshots in range(0, 6):
            numshots = 2 ** numshots
            for numqueries in [100]:

                distractor = np.random.choice([c for c in allclasses if c != held_out])
                distractor1 = np.random.choice([c for c in allclasses if c != held_out and c!= distractor])
                # held_out_classes = [held_out, distractor]  # these are what will be compared
                held_out_classes = [held_out, distractor, distractor1]  # these are what will be compared
                held_out_classes = allclasses
                support_x, support_y, query_x, query_y = load_episode_1d(data_dir, held_out_classes, shot=numshots, query_per_class=numqueries)

                start = time.time()
                ############################
                with torch.no_grad():
                    support_embed = encoder(support_x)
                    query_embed = encoder(query_x)
                    # TESTING (MEASURE THIS)
                    prototypes, classes = compute_prototypes(support_embed, support_y)
                    dists = classify_queries(query_embed, prototypes)
                ############################
                end = time.time()
                elapsed_ms = (end - start) * 1000

                preds = torch.argmin(dists, dim=1)
                classes = torch.unique(support_y)
                true = torch.tensor([classes.tolist().index(label.item()) for label in query_y])
                acc = (preds == true).float().mean().item()

                target_index = held_out_classes.index(held_out)
                mask = true == target_index
                acc = (preds[mask] == true[mask]).float().mean().item()

                # acc = max(acc, maxacc)
                # maxacc = acc

                print(held_out, numqueries, numshots, acc, elapsed_ms)
                dflist.append({'numepisodes': episodes, 'numqueries': numqueries, 'numshots': numshots, 'acc': acc, 'elapsed_ms': int(elapsed_ms), "held_out": held_out})

            df = pd.DataFrame(dflist)
            pivot_table = df.pivot_table(index="numqueries", columns="numshots", values="acc", aggfunc="mean")
            pivot_table.round(2).to_csv("queries_vs_shots" + way + ".csv")
            pivot_table = df.pivot_table(index="held_out", columns="numshots", values="acc", aggfunc="mean")
            pivot_table.round(2).to_csv("heldout_vs_shots" + way + ".csv")
            pivot_table = df.pivot_table(index="numqueries", columns="numshots", values="elapsed_ms", aggfunc="mean")
            pivot_table.round(2).to_csv("queries_vs_shots_time" + way + ".csv")


if __name__ == "__main__":
    classes = ['lefturn', 'rightturn', 'ra', 'lanechange', "lefttoright", "righttoleft", "stopinmiddle", "waittocross"]
    # evaluate_all_held_out(data_dir="../data/data_csv", episodes=5, eval_repeats=5, classes=classes)
    evaluate_acc_time(data_dir="../data/data_csv")

