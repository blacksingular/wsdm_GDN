import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
"""
	GDN Model
"""


class GDNLayer(nn.Module):
	"""
	One GDN layer
	"""

	def __init__(self, num_classes, inter1):
		"""
		Initialize GDN model
		:param num_classes: number of classes (2 in our paper)
		:param inter1: the inter-relation aggregator that output the final embedding
		"""
		super(GDNLayer, self).__init__()
		self.inter1 = inter1
		self.xent = nn.CrossEntropyLoss()
		self.softmax = nn.Softmax(dim=-1)
		self.KLDiv = nn.KLDivLoss(reduction='batchmean')
		self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
		# the parameter to transform the final embedding
		self.weight = nn.Parameter(torch.FloatTensor(num_classes, inter1.embed_dim))
		init.xavier_uniform_(self.weight)


	def forward(self, nodes, labels):
		embeds1 = self.inter1(nodes, labels)
		scores = self.weight.mm(embeds1)
		return scores.t()

	def to_prob(self, nodes, labels):
		gnn_logits = self.forward(nodes, labels)
		gnn_scores = self.softmax(gnn_logits)
		return gnn_scores

	def loss(self, nodes, labels):
		gnn_scores = self.forward(nodes, labels)
		# GNN loss
		gnn_loss = self.xent(gnn_scores, labels.squeeze())
		return gnn_loss
	