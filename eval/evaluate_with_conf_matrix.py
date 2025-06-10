import torch
import torch.optim as optim
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from model.encoder import TinyCNNEncoder
from train.train_fewshot import train_one_episode
from model.proto_net import compute_prototypes, classify_queries
from utils.loader import load_episode_custom


def log_confusion_matrix(preds, true, class_labels, save_path="confusion_matrix.png"):
    cm = confusion_matrix(true.numpy(), preds.numpy(), labels=range(len(class_labels)))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()


def evaluate_with_conf_matrix(data_dir="data_csv", episodes=10):
    encoder = TinyCNNEncoder()
    optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    included_classes = ['normal', 'spike', 'drop', 'pattern']

    for episode in range(episodes):
        data = load_episode_custom(data_dir, included_classes)
        train_one_episode(encoder, data, optimizer)

    # Evaluate final performance on a new episode
    support_x, support_y, query_x, query_y = load_episode_custom(data_dir, included_classes)
    with torch.no_grad():
        support_embed = encoder(support_x)
        query_embed = encoder(query_x)
        prototypes, classes = compute_prototypes(support_embed, support_y)
        dists = classify_queries(query_embed, prototypes)
        preds = torch.argmin(dists, dim=1)
        true = torch.tensor([classes.tolist().index(label.item()) for label in query_y])
        all_class_names = ['normal', 'spike', 'drop', 'pattern']
        label_names = [all_class_names[i] for i in classes.tolist()]
        log_confusion_matrix(preds, true, class_labels=label_names, save_path="conf_matrix_eval.png")


if __name__ == "__main__":
    evaluate_with_conf_matrix(data_dir="../data/data_csv")