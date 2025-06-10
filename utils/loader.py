import torch
import numpy as np
import os
import pandas as pd


def load_episode_custom(data_dir, included_classes, shot=5, query_per_class=5):
    support_x, support_y, query_x, query_y = [], [], [], []

    for idx, cls in enumerate(included_classes):
        files = [f for f in os.listdir(data_dir) if cls in f and f.endswith('.csv')]
        np.random.shuffle(files)
        support = files[:shot]  # a few (shot) labeled examples per class
        query = files[shot:shot + query_per_class]  # query_per_class different examples from the same class(es) used for classification

        for f in support:
            df = pd.read_csv(os.path.join(data_dir, f))
            x = df['value'].values
            support_x.append(x)
            support_y.append(idx)

        for f in query:
            df = pd.read_csv(os.path.join(data_dir, f))
            x = df['value'].values
            query_x.append(x)
            query_y.append(idx)

    support_x = torch.tensor(support_x).float().unsqueeze(-1)
    support_y = torch.tensor(support_y)
    query_x = torch.tensor(query_x).float().unsqueeze(-1)
    query_y = torch.tensor(query_y)

    return support_x, support_y, query_x, query_y