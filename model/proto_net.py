import torch

# few-shot prototypical network
def compute_prototypes(support_embeddings, support_labels):
    classes = torch.unique(support_labels)
    prototypes = []
    for c in classes:
        class_embed = support_embeddings[support_labels == c]
        prototypes.append(class_embed.mean(dim=0))
    return torch.stack(prototypes), classes

def classify_queries(query_embeddings, prototypes):
    dists = torch.cdist(query_embeddings, prototypes)
    return dists