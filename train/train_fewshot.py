import torch
import torch.nn.functional as F
from model.proto_net import compute_prototypes, classify_queries

# single-episode training logic
def train_one_episode(encoder, data, optimizer):
	# encoded support and query examples
	support_x, support_y, query_x, query_y = data
	support_embed = encoder(support_x)
	query_embed = encoder(query_x)

	# build prototypes and retrieve unique class order
	prototypes, classes = compute_prototypes(support_embed, support_y)

	# classify queries by distance to prototypes
	dists = classify_queries(query_embed, prototypes)
	preds = torch.argmin(dists, dim=1)

	# true query labels to indices in prototype order
	true = torch.tensor([classes.tolist().index(label.item()) for label in query_y])

	# print("True labels:", true.tolist())
	# print("Predictions:", preds.tolist())

	loss = F.cross_entropy(-dists, true)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()  # updating the encoder weights, better at forming embeddings that separate classes over episodes

	return loss.item(), (preds == true).float().mean().item()