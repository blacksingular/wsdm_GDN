import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import random
import pdb


"""
	GraphSAGE implementations
	Paper: Inductive Representation Learning on Large Graphs
	Source: https://github.com/williamleif/graphsage-simple/
"""


class GraphSage(nn.Module):
	"""
	Vanilla GraphSAGE Model
	Code partially from https://github.com/williamleif/graphsage-simple/
	"""
	def __init__(self, num_classes, enc):
		super(GraphSage, self).__init__()
		self.enc = enc
		self.xent = nn.CrossEntropyLoss()
		self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
		init.xavier_uniform_(self.weight)

	def forward(self, nodes):
		embeds = self.enc(nodes)
		scores = self.weight.mm(embeds)
		return scores.t()

	def to_prob(self, nodes, train_flag=True):
		pos_scores = torch.sigmoid(self.forward(nodes))
		return pos_scores

	def loss(self, nodes, labels):
		scores = self.forward(nodes)
		return self.xent(scores, labels.squeeze())
	

class MeanAggregator(nn.Module):
	"""
	Aggregates a node's embeddings using mean of neighbors' embeddings
	"""

	def __init__(self, features, cuda=False, gcn=False):
		"""
		Initializes the aggregator for a specific graph.

		features -- function mapping LongTensor of node ids to FloatTensor of feature values.
		cuda -- whether to use GPU
		gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
		"""

		super(MeanAggregator, self).__init__()

		self.features = features
		self.cuda = cuda
		self.gcn = gcn

	def forward(self, nodes, to_neighs, num_sample=10):
		"""
		nodes --- list of nodes in a batch
		to_neighs --- list of sets, each set is the set of neighbors for node in batch
		num_sample --- number of neighbors to sample. No sampling if None.
		"""
		# Local pointers to functions (speed hack)
		_set = set
		if not num_sample is None:
			_sample = random.sample
			samp_neighs = [_set(_sample(to_neigh,
										num_sample,
										)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
		else:
			samp_neighs = to_neighs

		if self.gcn:
			samp_neighs = [samp_neigh.union(set([int(nodes[i])])) for i, samp_neigh in enumerate(samp_neighs)]
		unique_nodes_list = list(set.union(*samp_neighs))
		unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
		mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
		column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
		row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
		mask[row_indices, column_indices] = 1
		if self.cuda:
			mask = mask.cuda()
		num_neigh = mask.sum(1, keepdim=True)
		mask = mask.div(num_neigh)
		if self.cuda:
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
		else:
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
		to_feats = mask.mm(embed_matrix)
		return to_feats


class Encoder(nn.Module):
	"""
	Vanilla GraphSAGE Encoder Module
	Encodes a node's using 'convolutional' GraphSage approach
	"""
	def __init__(self, features, feature_dim,
				 embed_dim, adj_lists, aggregator,
				 train_pos, train_neg, num_sample=10, 
				 base_model=None, gcn=False, cuda=False,
				 feature_transform=False):
		super(Encoder, self).__init__()

		self.features = features
		self.feat_dim = feature_dim
		self.adj_lists = adj_lists
		self.aggregator = aggregator
		self.num_sample = num_sample
		if base_model != None:
			self.base_model = base_model

		self.gcn = gcn
		self.embed_dim = embed_dim
		self.cuda = cuda
		self.aggregator.cuda = cuda
		self.weight = nn.Parameter(
			torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
		init.xavier_uniform_(self.weight)

		self.pos_vector = None
		self.neg_vector = None
		self.train_pos = train_pos
		self.train_neg = train_neg
		self.softmax = nn.Softmax(dim=-1)
		self.KLDiv = nn.KLDivLoss(reduction='batchmean')
		self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

		if self.cuda and isinstance(self.train_pos, list) and isinstance(self.train_neg, list):
			self.pos_index = torch.LongTensor(self.train_pos).cuda()
			self.neg_index = torch.LongTensor(self.train_neg).cuda()
		else:
			self.pos_index = torch.LongTensor(self.train_pos)
			self.neg_index = torch.LongTensor(self.train_neg)

		self.unique_nodes = set()
		for _, adj_list in self.adj_lists.items():
			self.unique_nodes = self.unique_nodes.union(adj_list)
	# def __init__(self, features, feature_dim,
	# 			 embed_dim, adj_lists, aggregator,
	# 			 num_sample=10,
	# 			 base_model=None, gcn=False, cuda=False,
	# 			 feature_transform=False):
	# 	super(Encoder, self).__init__()

	# 	self.features = features
	# 	self.feat_dim = feature_dim
	# 	self.adj_lists = adj_lists
	# 	self.aggregator = aggregator
	# 	self.num_sample = num_sample
	# 	if base_model != None:
	# 		self.base_model = base_model

	# 	self.gcn = gcn
	# 	self.embed_dim = embed_dim
	# 	self.cuda = cuda
	# 	self.aggregator.cuda = cuda
	# 	self.weight = nn.Parameter(
	# 		torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
	# 	init.xavier_uniform_(self.weight)

	def forward(self, nodes):
		"""
		Generates embeddings for a batch of nodes.

		nodes     -- list of nodes
		"""
		neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes],
											  self.num_sample)
		self.update_label_vector(self.features)
		if isinstance(nodes, list):
			index = torch.LongTensor(nodes)
		else:
			index = nodes

		if not self.gcn:
			if self.cuda:
				self_feats = self.features(index).cuda()
			else:
				self_feats = self.features(index)
			combined = torch.cat((self_feats, neigh_feats), dim=1)
		else:
			combined = neigh_feats
		combined = F.relu(self.weight.mm(combined.t()))
		return combined

	def constraint_loss(self, grads_idx, with_grad):
		if with_grad:
			x = F.log_softmax(self.features(self.pos_index)[:, grads_idx], dim=-1)
			# x = F.log_softmax(self.stored_hidden[self.train_pos][:, grads_idx], dim=-1)
			target_pos = self.pos_vector[:, grads_idx].repeat(x.shape[0], 1).softmax(dim=-1)
			target_neg = self.neg_vector[:, grads_idx].repeat(x.shape[0], 1).softmax(dim=-1)
			loss_pos = self.KLDiv(x, target_pos)
			loss_neg = self.KLDiv(x, target_neg)
			# pdb.set_trace()
		else:
			x = F.log_softmax(self.features(self.pos_index), dim=-1)
			# x = F.log_softmax(self.stored_hidden[self.train_pos], dim=-1)
			target_pos = self.pos_vector.repeat(x.shape[0], 1).softmax(dim=-1)
			target_neg = self.neg_vector.repeat(x.shape[0], 1).softmax(dim=-1)
			loss_pos = self.KLDiv(x, target_pos)
			loss_neg = self.KLDiv(x, target_neg)
			# pdb.set_trace()
		return loss_pos, loss_neg
	
	def softmax_with_temperature(self, input, t=1, axis=-1):
		ex = torch.exp(input/t)
		sum = torch.sum(ex, axis=axis)
		return ex/sum

	def fn_loss(self, nodes, non_grad_idx):
		pos_nodes = set(self.train_pos)
		target = []
		neighs = []
		for node in nodes:
			if int(node) in pos_nodes:
				target.append(int(node))
				neighs.append(self.adj_lists[int(node)])
		x = F.log_softmax(self.fetch_feat(target)[:, non_grad_idx], dim=-1)
		pos = torch.zeros_like(self.fetch_feat(target))
		neg = torch.zeros_like(self.fetch_feat(target))
		for i in range(len(target)):
			pos[i] = torch.mean(self.fetch_feat(list(neighs[i])), dim=0, keepdim=True)
			neg_idx = [random.choice(list(self.unique_nodes.difference(neighs[i])))]
			neg[i] = self.fetch_feat(neg_idx)
			# pdb.set_trace()
		pos = pos[:, non_grad_idx].softmax(dim=-1)
		neg = neg[:, non_grad_idx].softmax(dim=-1)
		loss_pos = self.KLDiv(x, pos)
		loss_neg = self.KLDiv(x, neg)
		# pdb.set_trace()
		return loss_pos, loss_neg

	def fetch_feat(self, nodes):
		if self.cuda and isinstance(nodes, list):
			index = torch.LongTensor(nodes).cuda()
		else:
			index = torch.LongTensor(nodes)
		return self.features(index)

	def update_label_vector(self, x):
		# pdb.set_trace()
		if isinstance(x, torch.Tensor):
			x_pos = x[self.train_pos]
			x_neg = x[self.train_neg]
		elif isinstance(x, torch.nn.Embedding):
			x_pos = x(self.pos_index)
			x_neg = x(self.neg_index)
		if self.pos_vector is None:
			self.pos_vector = torch.mean(x_pos, dim=0, keepdim=True).detach()
			self.neg_vector = torch.mean(x_neg, dim=0, keepdim=True).detach()
		else:
			cosine_pos = self.cos(self.pos_vector, x_pos)
			cosine_neg = self.cos(self.neg_vector, x_neg)
			weights_pos = self.softmax_with_temperature(cosine_pos, t=5).reshape(1, -1)
			weights_neg = self.softmax_with_temperature(cosine_neg, t=5).reshape(1, -1)
			self.pos_vector = torch.mm(weights_pos, x_pos).detach()
			self.neg_vector = torch.mm(weights_neg, x_neg).detach()



class GCN(nn.Module):
	"""
	Vanilla GCN Model
	Code partially from https://github.com/williamleif/graphsage-simple/
	"""
	def __init__(self, num_classes, enc, add_constraint):
		super(GCN, self).__init__()
		self.enc = enc
		self.xent = nn.CrossEntropyLoss()
		self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
		init.xavier_uniform_(self.weight)
		self.pos_vector = None
		self.neg_vector = None


	def forward(self, nodes, train_flag=True):
		if train_flag:
			embeds = self.enc(nodes, train_flag)
			scores = self.weight.mm(embeds)
		else:
			embeds = self.enc(nodes, train_flag)
			scores = self.weight.mm(embeds)
		return scores.t()

	def to_prob(self, nodes, train_flag=True):
		pos_scores = torch.sigmoid(self.forward(nodes, train_flag))
		return pos_scores

	def loss(self, nodes, labels):
		scores = self.forward(nodes)
		return self.xent(scores, labels.squeeze())


class GCNAggregator(nn.Module):
	"""
	Aggregates a node's embeddings using normalized mean of neighbors' embeddings
	"""

	def __init__(self, features, cuda=False, gcn=False):
		"""
		Initializes the aggregator for a specific graph.

		features -- function mapping LongTensor of node ids to FloatTensor of feature values.
		cuda -- whether to use GPU
		gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
		"""

		super(GCNAggregator, self).__init__()

		self.features = features
		self.cuda = cuda
		self.gcn = gcn

	def forward(self, nodes, to_neighs):
		"""
		nodes --- list of nodes in a batch
		to_neighs --- list of sets, each set is the set of neighbors for node in batch
		"""
		# Local pointers to functions (speed hack)
		
		samp_neighs = to_neighs

		#  Add self to neighs
		samp_neighs = [samp_neigh.union(set([int(nodes[i])])) for i, samp_neigh in enumerate(samp_neighs)]
		unique_nodes_list = list(set.union(*samp_neighs))
		unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
		mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
		column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
		row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
		mask[row_indices, column_indices] = 1.0  # Adjacency matrix for the sub-graph
		if self.cuda:
			mask = mask.cuda()
		row_normalized = mask.sum(1, keepdim=True).sqrt()
		col_normalized = mask.sum(0, keepdim=True).sqrt()
		mask = mask.div(row_normalized).div(col_normalized)
		if self.cuda:
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
		else:
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
		to_feats = mask.mm(embed_matrix)
		return to_feats

class GCNEncoder(nn.Module):
	"""
	GCN Encoder Module
	"""

	def __init__(self, features, feature_dim,
				 embed_dim, adj_lists, aggregator,
				 train_pos, train_neg, num_sample=10, 
				 base_model=None, gcn=False, cuda=False,
				 feature_transform=False):
		super(GCNEncoder, self).__init__()

		self.features = features
		self.feat_dim = feature_dim
		self.adj_lists = adj_lists
		self.aggregator = aggregator
		self.num_sample = num_sample
		if base_model != None:
			self.base_model = base_model

		self.gcn = gcn
		self.embed_dim = embed_dim
		self.cuda = cuda
		self.aggregator.cuda = cuda
		self.weight = nn.Parameter(
			torch.FloatTensor(embed_dim, self.feat_dim ))
		init.xavier_uniform_(self.weight)

		self.pos_vector = None
		self.neg_vector = None
		self.train_pos = train_pos
		self.train_neg = train_neg
		self.softmax = nn.Softmax(dim=-1)
		self.KLDiv = nn.KLDivLoss(reduction='batchmean')
		self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

		if self.cuda and isinstance(self.train_pos, list) and isinstance(self.train_neg, list):
			self.pos_index = torch.LongTensor(self.train_pos).cuda()
			self.neg_index = torch.LongTensor(self.train_neg).cuda()
		else:
			self.pos_index = torch.LongTensor(self.train_pos)
			self.neg_index = torch.LongTensor(self.train_neg)

		self.unique_nodes = set()
		for _, adj_list in self.adj_lists.items():
			self.unique_nodes = self.unique_nodes.union(adj_list)

		

	def forward(self, nodes, train_flag=True):
		"""
		Generates embeddings for a batch of nodes.
		Input:
			nodes -- list of nodes
		Output:
		    embed_dim*len(nodes)
		"""
		neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes])
		self.update_label_vector(self.features)
		if isinstance(nodes, list):
			index = torch.LongTensor(nodes)
		else:
			index = nodes
		self_feats = self.features(index)

		combined = F.relu(self.weight.mm(neigh_feats.t()))
		if not train_flag:
			cosine_pos = self.cos(self.pos_vector, self_feats).detach()
			cosine_neg = self.cos(self.neg_vector, self_feats).detach()
			simi_scores = torch.cat((cosine_neg.view(-1, 1), cosine_pos.view(-1, 1)), 1).t()
			# simi_scores = torch.zeros(2, self_feats.shape[0])
			return combined, simi_scores
		return combined

	def update_label_vector(self, x):
		# pdb.set_trace()
		if isinstance(x, torch.Tensor):
			x_pos = x[self.train_pos]
			x_neg = x[self.train_neg]
		elif isinstance(x, torch.nn.Embedding):
			x_pos = x(self.pos_index)
			x_neg = x(self.neg_index)
		if self.pos_vector is None:
			self.pos_vector = torch.mean(x_pos, dim=0, keepdim=True).detach()
			self.neg_vector = torch.mean(x_neg, dim=0, keepdim=True).detach()
		else:
			cosine_pos = self.cos(self.pos_vector, x_pos)
			cosine_neg = self.cos(self.neg_vector, x_neg)
			weights_pos = self.softmax_with_temperature(cosine_pos, t=5).reshape(1, -1)
			weights_neg = self.softmax_with_temperature(cosine_neg, t=5).reshape(1, -1)
			self.pos_vector = torch.mm(weights_pos, x_pos).detach()
			self.neg_vector = torch.mm(weights_neg, x_neg).detach()
		# pdb.set_trace()

	def constraint_loss(self, grads_idx, with_grad):
		if with_grad:
			x = F.log_softmax(self.features(self.pos_index)[:, grads_idx], dim=-1)
			# x = F.log_softmax(self.stored_hidden[self.train_pos][:, grads_idx], dim=-1)
			target_pos = self.pos_vector[:, grads_idx].repeat(x.shape[0], 1).softmax(dim=-1)
			target_neg = self.neg_vector[:, grads_idx].repeat(x.shape[0], 1).softmax(dim=-1)
			loss_pos = self.KLDiv(x, target_pos)
			loss_neg = self.KLDiv(x, target_neg)
			# pdb.set_trace()
		else:
			x = F.log_softmax(self.features(self.pos_index), dim=-1)
			# x = F.log_softmax(self.stored_hidden[self.train_pos], dim=-1)
			target_pos = self.pos_vector.repeat(x.shape[0], 1).softmax(dim=-1)
			target_neg = self.neg_vector.repeat(x.shape[0], 1).softmax(dim=-1)
			loss_pos = self.KLDiv(x, target_pos)
			loss_neg = self.KLDiv(x, target_neg)
			# pdb.set_trace()
		return loss_pos, loss_neg
	
	def softmax_with_temperature(self, input, t=1, axis=-1):
		ex = torch.exp(input/t)
		sum = torch.sum(ex, axis=axis)
		return ex/sum

	def fn_loss(self, nodes, non_grad_idx):
		pos_nodes = set(self.train_pos)
		target = []
		neighs = []
		for node in nodes:
			if int(node) in pos_nodes:
				target.append(int(node))
				neighs.append(self.adj_lists[int(node)])
		x = F.log_softmax(self.fetch_feat(target)[:, non_grad_idx], dim=-1)
		pos = torch.zeros_like(self.fetch_feat(target))
		neg = torch.zeros_like(self.fetch_feat(target))
		for i in range(len(target)):
			pos[i] = torch.mean(self.fetch_feat(list(neighs[i])), dim=0, keepdim=True)
			neg_idx = [random.choice(list(self.unique_nodes.difference(neighs[i])))]
			neg[i] = self.fetch_feat(neg_idx)
			# pdb.set_trace()
		pos = pos[:, non_grad_idx].softmax(dim=-1)
		neg = neg[:, non_grad_idx].softmax(dim=-1)
		loss_pos = self.KLDiv(x, pos)
		loss_neg = self.KLDiv(x, neg)
		# pdb.set_trace()
		return loss_pos, loss_neg

	def fetch_feat(self, nodes):
		if self.cuda and isinstance(nodes, list):
			index = torch.LongTensor(nodes).cuda()
		else:
			index = torch.LongTensor(nodes)
		return self.features(index)
