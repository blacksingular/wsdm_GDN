import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import random

"""
	GDN Layers
"""


class InterAgg(nn.Module):

	def __init__(self, features, feature_dim, embed_dim, 
				 train_pos, train_neg, adj_lists, intraggs, inter='GNN', cuda=True):
		"""
		Initialize the inter-relation aggregator
		:param features: the input node features or embeddings for all nodes
		:param feature_dim: the input dimension
		:param embed_dim: the embed dimension
		:param train_pos: positive samples in training set
		:param adj_lists: a list of adjacency lists for each single-relation graph
		:param intraggs: the intra-relation aggregators used by each single-relation graph
		:param inter: NOT used in this version, the aggregator type: 'Att', 'Weight', 'Mean', 'GNN'
		:param cuda: whether to use GPU
		"""
		super(InterAgg, self).__init__()

		# stored parameters
		self.features = features
		self.pos_vector = None
		self.neg_vector = None

		# Functions
		self.softmax = nn.Softmax(dim=-1)
		self.KLDiv = nn.KLDivLoss(reduction='batchmean')
		self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

		self.dropout = 0.6
		self.adj_lists = adj_lists
		self.intra_agg1 = intraggs[0]
		self.intra_agg2 = intraggs[1]
		self.intra_agg3 = intraggs[2]
		self.embed_dim = embed_dim
		self.feat_dim = feature_dim
		self.inter = inter
		self.cuda = cuda
		self.intra_agg1.cuda = cuda
		self.intra_agg2.cuda = cuda
		self.intra_agg3.cuda = cuda
		self.train_pos = train_pos
		self.train_neg = train_neg

		# initial filtering thresholds
		self.thresholds = [0.5, 0.5, 0.5]

		# parameter used to transform node embeddings before inter-relation aggregation
		self.weight = nn.Parameter(torch.FloatTensor(self.embed_dim*len(intraggs)+self.feat_dim, self.embed_dim))
		init.xavier_uniform_(self.weight)

		# label predictor for similarity measure
		self.label_clf = nn.Linear(self.feat_dim, 2)

		# initialize the parameter logs
		self.weights_log = []
		self.thresholds_log = [self.thresholds]
		self.relation_score_log = []


		if self.cuda and isinstance(self.train_pos, list) and isinstance(self.train_neg, list):
			self.pos_index = torch.LongTensor(self.train_pos).cuda()
			self.neg_index = torch.LongTensor(self.train_neg).cuda()
		else:
			self.pos_index = torch.LongTensor(self.train_pos)
			self.neg_index = torch.LongTensor(self.train_neg)

	def forward(self, nodes, labels):
		"""
		:param nodes: a list of batch node ids
		:param labels: a list of batch node labels
		:param train_flag: indicates whether in training or testing mode
		:return combined: the embeddings of a batch of input node features
		:return center_scores: the label-aware scores of batch nodes
		"""

		# extract 1-hop neighbor ids from adj lists of each single-relation graph
		to_neighs = []
		for adj_list in self.adj_lists:
			to_neighs.append([set(adj_list[int(node)]) for node in nodes])

		# get neighbor node id list for each batch node and relation
		r1_list = [list(to_neigh) for to_neigh in to_neighs[0]]
		r2_list = [list(to_neigh) for to_neigh in to_neighs[1]]
		r3_list = [list(to_neigh) for to_neigh in to_neighs[2]]

		# find unique nodes and their neighbors used in current batch
		unique_nodes = set.union(set.union(*to_neighs[0]), set.union(*to_neighs[1]),
									set.union(*to_neighs[2], set(nodes)))
		self.unique_nodes = unique_nodes

		# intra-aggregation steps for each relation
		r1_feats = self.intra_agg1.forward(nodes, r1_list)
		r2_feats = self.intra_agg2.forward(nodes, r2_list)
		r3_feats = self.intra_agg3.forward(nodes, r3_list)

		# get features or embeddings for batch nodes
		self_feats = self.fetch_feat(nodes)
		
		# Update label vector
		self.update_label_vector(self.features)

		# concat the intra-aggregated embeddings from each relation
		cat_feats = torch.cat((self_feats, r1_feats, r2_feats, r3_feats), dim=1)
		combined = F.relu(cat_feats.mm(self.weight).t())
		return combined

	def fetch_feat(self, nodes):
		if self.cuda and isinstance(nodes, list):
			index = torch.LongTensor(nodes).cuda()
		else:
			index = torch.LongTensor(nodes)
		return self.features(index)

	def cal_simi_scores(self, nodes):
		self_feats = self.fetch_feat(nodes)
		cosine_pos = self.cos(self.pos_vector, self_feats).detach()
		cosine_neg = self.cos(self.neg_vector, self_feats).detach()
		simi_scores = torch.cat((cosine_neg.view(-1, 1), cosine_pos.view(-1, 1)), 1)
		return simi_scores

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

	def fl_loss(self, grads_idx):
		x = F.log_softmax(self.features(self.pos_index)[:, grads_idx], dim=-1)
		target_pos = self.pos_vector[:, grads_idx].repeat(x.shape[0], 1).softmax(dim=-1)
		target_neg = self.neg_vector[:, grads_idx].repeat(x.shape[0], 1).softmax(dim=-1)
		loss_pos = self.KLDiv(x, target_pos)
		loss_neg = self.KLDiv(x, target_neg)
		return loss_pos, loss_neg
	
	def fn_loss(self, nodes, non_grad_idx):
		pos_nodes = set(self.train_pos)
		to_neighs = []
		target = []
		for adj_list in self.adj_lists:
			target_r = []
			to_neighs_r = []
			for node in nodes:
				if int(node) in pos_nodes:
					target_r.append(int(node))
					to_neighs_r.append(set(adj_list[int(node)]))
			to_neighs.append(to_neighs_r)
			target.append(target_r)
		
		to_neighs_all = []
		for x, y, z in zip(to_neighs[0], to_neighs[1], to_neighs[2]):
			to_neighs_all.append(set.union(x, y, z))

		r1_list = [list(to_neigh) for to_neigh in to_neighs[0]]
		r2_list = [list(to_neigh) for to_neigh in to_neighs[1]]
		r3_list = [list(to_neigh) for to_neigh in to_neighs[2]]
		# print(non_grad_idx)
		pos_1, neg_1 = self.intra_agg1.fn_loss(non_grad_idx, target[0], r1_list, self.unique_nodes, to_neighs_all)
		pos_2, neg_2 = self.intra_agg2.fn_loss(non_grad_idx, target[1], r2_list, self.unique_nodes, to_neighs_all)
		pos_3, neg_3 = self.intra_agg3.fn_loss(non_grad_idx, target[2], r3_list, self.unique_nodes, to_neighs_all)
		return pos_1 + pos_2 + pos_3, neg_1 + neg_2 + neg_3
	def softmax_with_temperature(self, input, t=1, axis=-1):
		ex = torch.exp(input/t)
		sum = torch.sum(ex, axis=axis)
		return ex/sum


class IntraAgg(nn.Module):

	def __init__(self, features, feat_dim, embed_dim, train_pos, cuda=False):
		"""
		Initialize the intra-relation aggregator
		:param features: the input node features or embeddings for all nodes
		:param feat_dim: the input dimension
		:param embed_dim: the embed dimension
		:param train_pos: positive samples in training set
		:param cuda: whether to use GPU
		"""
		super(IntraAgg, self).__init__()

		self.features = features
		self.cuda = cuda
		self.feat_dim = feat_dim
		self.embed_dim = embed_dim
		self.train_pos = train_pos
		self.weight = nn.Parameter(torch.FloatTensor(2*self.feat_dim, self.embed_dim))
		init.xavier_uniform_(self.weight)

		self.KLDiv = nn.KLDivLoss(reduction='batchmean')

	def forward(self, nodes, to_neighs_list):
		"""
		Code partially from https://github.com/williamleif/graphsage-simple/
		:param nodes: list of nodes in a batch
		:param to_neighs_list: neighbor node id list for each batch node in one relation
		:param batch_scores: the label-aware scores of batch nodes
		:param neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation
		:param pos_scores: the label-aware scores 1-hop neighbors for the minority positive nodes
		:param train_flag: indicates whether in training or testing mode
		:param sample_list: the number of neighbors kept for each batch node in one relation
		:return to_feats: the aggregated embeddings of batch nodes neighbors in one relation
		:return samp_scores: the average neighbor distances for each relation after filtering
		"""

		samp_neighs = [set(x) for x in to_neighs_list]
		# find the unique nodes among batch nodes and the filtered neighbors
		unique_nodes_list = list(set.union(*samp_neighs))
		unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
		

		# intra-relation aggregation only with sampled neighbors
		mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
		column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
		row_indices = [i for i in range(len(samp_neighs)) for _ in range(len(samp_neighs[i]))]
		mask[row_indices, column_indices] = 1
		if self.cuda:
			mask = mask.cuda()
		num_neigh = mask.sum(1, keepdim=True)
		mask = mask.div(num_neigh)  # mean aggregator
		if self.cuda:
			self_feats = self.features(torch.LongTensor(nodes).cuda())
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
		else:
			self_feats = self.features(torch.LongTensor(nodes))
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
		agg_feats = mask.mm(embed_matrix)  # single relation aggregator
		cat_feats = torch.cat((self_feats, agg_feats), dim=1)  # concat with last layer
		to_feats = F.relu(cat_feats.mm(self.weight))
		return to_feats

	def update_label_vector(self, x):
		# pdb.set_trace()
		if self.cuda and isinstance(self.train_pos, list) and isinstance(self.train_neg, list):
			pos_index = torch.LongTensor(self.train_pos).cuda()
			neg_index = torch.LongTensor(self.train_neg).cuda()
		if self.pos_vector is None:
			self.pos_vector = torch.mean(x(pos_index), dim=0, keepdim=True).detach()
			self.neg_vector = torch.mean(x(neg_index), dim=0, keepdim=True).detach()
		else:
			cosine_pos = self.cos(self.pos_vector, x(pos_index))
			cosine_neg = self.cos(self.neg_vector, x(neg_index))
			weights_pos = self.softmax_with_temperature(cosine_pos, t=5).reshape(1, -1)
			weights_neg = self.softmax_with_temperature(cosine_neg, t=5).reshape(1, -1)
			self.pos_vector = torch.mm(weights_pos, x(pos_index)).detach()
			self.neg_vector = torch.mm(weights_neg, x(neg_index)).detach()

	def fetch_feat(self, nodes):
		if self.cuda and isinstance(nodes, list):
			index = torch.LongTensor(nodes).cuda()
		else:
			index = torch.LongTensor(nodes)
		return self.features(index)

	def softmax_with_temperature(self, input, t=1, axis=-1):
		ex = torch.exp(input/t)
		sum = torch.sum(ex, axis=axis)
		return ex/sum
	
	def fn_loss(self, non_grad_idx, target, neighs, all_nodes, all_neighs):
		x = F.log_softmax(self.fetch_feat(target)[:, non_grad_idx], dim=-1)
		pos = torch.zeros_like(self.fetch_feat(target))
		neg = torch.zeros_like(self.fetch_feat(target))
		for i in range(len(target)):
			pos[i] = torch.mean(self.fetch_feat(neighs[i]), dim=0, keepdim=True)
			neg_idx = [random.choice(list(all_nodes.difference(all_neighs[i])))]
			neg[i] = self.fetch_feat(neg_idx)
			# pdb.set_trace()
		pos = pos[:, non_grad_idx].softmax(dim=-1)
		neg = neg[:, non_grad_idx].softmax(dim=-1)
		loss_pos = self.KLDiv(x, pos)
		loss_neg = self.KLDiv(x, neg)
		# pdb.set_trace()
		return loss_pos, loss_neg