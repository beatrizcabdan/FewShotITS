import torch
import torch.optim as optim
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from model.encoder import TinyMLEncoder
from train.train_fewshot import train_one_episode
from model.proto_net import compute_prototypes, classify_queries
from train.episodic_loader import load_episode_1d
import time

def log_confusion_matrix(preds, true, class_labels, save_path="confusion_matrix.png"):
    cm = confusion_matrix(true.numpy(), preds.numpy(), labels=range(len(class_labels)))

    plt.figure(figsize=(6, 5))
    res = sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    res.axhline(y=0, color='k', linewidth=1)
    res.axhline(y=4, color='k', linewidth=2)
    res.axvline(x=0, color='k', linewidth=1)
    res.axvline(x=4, color='k', linewidth=2)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()

# shot: support examples per class
# query examples per class used to evaluate the classification accuracy
def evaluate_with_conf_matrix(data_dir="../data/data_csv", test_dir="../data/test_csv", episodes=10, shot=5, query_per_class=100, id=0):
    encoder = TinyMLEncoder()
    optimizer = optim.Adam(encoder.parameters(), lr=0.01)
    # included_classes = ['lefturn', 'rightturn', 'ra', 'lanechange']
    included_classes = ["lefttoright", "righttoleft", "stopinmiddle", "waittocross"]
    test_dir = data_dir

    # episodes are used to train the CNN encoder in few-shot learning pipeline
    for episode in range(episodes):
        data = load_episode_1d(data_dir, included_classes, shot=shot, query_per_class=query_per_class)
        train_one_episode(encoder, data, optimizer)

    # Evaluate final performance on a new episode
    support_x, support_y, query_x, query_y = load_episode_1d(test_dir, included_classes, shot=shot, query_per_class=query_per_class)
    with torch.no_grad():
        support_embed = encoder(support_x)
        query_embed = encoder(query_x)
        prototypes, classes = compute_prototypes(support_embed, support_y)
        dists = classify_queries(query_embed, prototypes)
        preds = torch.argmin(dists, dim=1)
        true = torch.tensor([classes.tolist().index(label.item()) for label in query_y])
        label_names = [included_classes[i] for i in classes.tolist()]
        log_confusion_matrix(preds, true, class_labels=label_names, save_path="evaluate_with_conf_matrix.png")

if __name__ == "__main__":
    evaluate_with_conf_matrix(data_dir="../data/data_csv")
    # evaluate_acc_time(data_dir="../data/data_csv")

