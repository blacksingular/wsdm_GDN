import pickle
import random
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
import copy as cp
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from collections import defaultdict
import logging


logging.basicConfig(filename='result.log',level=logging.INFO)
"""
	Utility functions to handle data and evaluate model.
"""

def biased_split(dataset):
	prefix = '/data'
	if dataset == 'yelp':
		data_name = 'YelpChi.mat'  # 'Amazon.mat' or 'YelpChi.mat'
	elif dataset == 'amazon':
		data_name = 'Amazon.mat'
	data = loadmat(prefix + data_name)

	if data_name == 'YelpChi.mat':
		net_list = [data['net_rur'].nonzero(), data['net_rtr'].nonzero(),
					data['net_rsr'].nonzero(), data['homo'].nonzero()]
	else:  # amazon dataset
		net_list = [data['net_upu'].nonzero(), data['net_usu'].nonzero(),
					data['net_uvu'].nonzero(), data['homo'].nonzero()]

	label = data['label'][0]
	pos_nodes = set(label.nonzero()[0].tolist())

	pos_node_dict, neg_node_dict = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
	# extract the edges of positive nodes in each relation graph
	net = net_list[-1]

	# calculate (homophily) probability for nodes
	for u, v in zip(net[0].tolist(), net[1].tolist()):
		if dataset == 'amazon' and min(u, v) < 3305:  # 0~3304 are unlabelled nodes for amazon
			continue
		if u in pos_nodes:
			pos_node_dict[u][0] += 1
			if label[u] == label[v]:
				pos_node_dict[u][1] += 1
		else:
			neg_node_dict[u][0] += 1
			if label[u] == label[v]:
				neg_node_dict[u][1] += 1

	p1 = np.zeros(len(label))
	for k in pos_node_dict:
		p1[k] = pos_node_dict[k][1] / pos_node_dict[k][0]
	p1 = p1 / p1.sum()
	pos_index = np.random.choice(range(len(p1)), size=round(0.6 * len(pos_node_dict)), replace=False, p = p1.ravel())
	p2 = np.zeros(len(label))
	for k in neg_node_dict:
		p2[k] = neg_node_dict[k][1] / neg_node_dict[k][0]
	p2 = p2 / p2.sum()
	neg_index = np.random.choice(range(len(p2)), size=round(0.6 * len(neg_node_dict)), replace=False, p = p2.ravel())

	idx_train = np.concatenate((pos_index, neg_index))
	np.random.shuffle(idx_train)
	idx_train = list(idx_train)
	y_train = np.array(label[idx_train])

	# find test label
	idx_test = list(set(range(len(label))).difference(set(idx_train)).difference(set(range(3305))))
	random.shuffle(idx_test)
	y_test = np.array(label[idx_test])
	return idx_train, idx_test, y_train, y_test
	
def load_data(data, prefix='/data'):
	"""
	Load graph, feature, and label given dataset name
	:returns: home and single-relation graphs, feature, label
	"""

	if data == 'yelp':
		data_file = loadmat(prefix + 'YelpChi.mat')
		labels = data_file['label'].flatten()
		feat_data = data_file['features'].todense().A
		# load the preprocessed adj_lists
		with open(prefix + 'yelp_homo_adjlists.pickle', 'rb') as file:
			homo = pickle.load(file)
		file.close()
		with open(prefix + 'yelp_rur_adjlists.pickle', 'rb') as file:
			relation1 = pickle.load(file)
		file.close()
		with open(prefix + 'yelp_rtr_adjlists.pickle', 'rb') as file:
			relation2 = pickle.load(file)
		file.close()
		with open(prefix + 'yelp_rsr_adjlists.pickle', 'rb') as file:
			relation3 = pickle.load(file)
		file.close()
	elif data == 'amazon':
		data_file = loadmat(prefix + 'Amazon.mat')
		labels = data_file['label'].flatten()
		feat_data = data_file['features'].todense().A
		# load the preprocessed adj_lists
		with open(prefix + 'amz_homo_adjlists.pickle', 'rb') as file:
			homo = pickle.load(file)
		file.close()
		with open(prefix + 'amz_upu_adjlists.pickle', 'rb') as file:
			relation1 = pickle.load(file)
		file.close()
		with open(prefix + 'amz_usu_adjlists.pickle', 'rb') as file:
			relation2 = pickle.load(file)
		file.close()
		with open(prefix + 'amz_uvu_adjlists.pickle', 'rb') as file:
			relation3 = pickle.load(file)

	return [homo, relation1, relation2, relation3], feat_data, labels


def normalize(mx):
	"""
		Row-normalize sparse matrix
		Code from https://github.com/williamleif/graphsage-simple/
	"""
	rowsum = np.array(mx.sum(1)) + 0.01
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	mx = r_mat_inv.dot(mx)
	return mx


def sparse_to_adjlist(sp_matrix, filename):
	"""
	Transfer sparse matrix to adjacency list
	:param sp_matrix: the sparse matrix
	:param filename: the filename of adjlist
	"""
	# add self loop
	homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
	# create adj_list
	adj_lists = defaultdict(set)
	edges = homo_adj.nonzero()
	for index, node in enumerate(edges[0]):
		adj_lists[node].add(edges[1][index])
		adj_lists[edges[1][index]].add(node)
	with open(filename, 'wb') as file:
		pickle.dump(adj_lists, file)
	file.close()


def pos_neg_split(nodes, labels):
	"""
	Find positive and negative nodes given a list of nodes and their labels
	:param nodes: a list of nodes
	:param labels: a list of node labels
	:returns: the spited positive and negative nodes
	"""
	pos_nodes = []
	neg_nodes = cp.deepcopy(nodes)
	aux_nodes = cp.deepcopy(nodes)
	for idx, label in enumerate(labels):
		if label == 1:
			pos_nodes.append(aux_nodes[idx])
			neg_nodes.remove(aux_nodes[idx])

	return pos_nodes, neg_nodes


def test_sage(test_cases, labels, model, batch_size, thres=0.5, save=False):
	"""
	Test the performance of GraphSAGE
	:param test_cases: a list of testing node
	:param labels: a list of testing node labels
	:param model: the GNN model
	:param batch_size: number nodes in a batch
	"""

	test_batch_num = int(len(test_cases) / batch_size) + 1
	gnn_pred_list = []
	gnn_prob_list = []
	for iteration in range(test_batch_num):
		i_start = iteration * batch_size
		i_end = min((iteration + 1) * batch_size, len(test_cases))
		batch_nodes = test_cases[i_start:i_end]
		gnn_prob = model.to_prob(batch_nodes, False)

		gnn_prob_arr = gnn_prob.data.cpu().numpy()[:, 1]
		gnn_pred = prob2pred(gnn_prob_arr, thres)
		
		gnn_pred_list.extend(gnn_pred.tolist())
		gnn_prob_list.extend(gnn_prob_arr.tolist())
	
	auc_gnn = roc_auc_score(labels, np.array(gnn_prob_list))
	f1_macro_gnn = f1_score(labels, np.array(gnn_pred_list), average='macro')
	conf_gnn = confusion_matrix(labels, np.array(gnn_pred_list))
	tn, fp, fn, tp = conf_gnn.ravel()
	gmean_gnn = conf_gmean(conf_gnn)

	logging.info(f"\tF1-macro: {f1_macro_gnn:.4f}\tG-Mean: {gmean_gnn:.4f}\tAUC: {auc_gnn:.4f}")
	logging.info(f"   GNN TP: {tp}\tTN: {tn}\tFN: {fn}\tFP: {fp}")
	return f1_macro_gnn, auc_gnn, gmean_gnn

	

def prob2pred(y_prob, thres=0.5):
	"""
	Convert probability to predicted results according to given threshold
	:param y_prob: numpy array of probability in [0, 1]
	:param thres: binary classification threshold, default 0.5
	:returns: the predicted result with the same shape as y_prob
	"""
	y_pred = np.zeros_like(y_prob, dtype=np.int32)
	y_pred[y_prob >= thres] = 1
	y_pred[y_prob < thres] = 0
	return y_pred


def test_GDN(test_cases, labels, model, batch_size, thres=0.5, save=False):
	"""
	Test the performance of GDN
	:param test_cases: a list of testing node
	:param labels: a list of testing node labels
	:param model: the GNN model
	:param batch_size: number nodes in a batch
	:returns: the AUC and Recall of GNN and Simi modules
	"""

	test_batch_num = int(len(test_cases) / batch_size) + 1
	gnn_pred_list = []
	gnn_prob_list = []

	for iteration in range(test_batch_num):
		i_start = iteration * batch_size
		i_end = min((iteration + 1) * batch_size, len(test_cases))
		batch_nodes = test_cases[i_start:i_end]
		batch_label = labels[i_start:i_end]
		gnn_prob = model.to_prob(batch_nodes, batch_label)
		gnn_prob_arr = gnn_prob.data.cpu().numpy()[:, 1]
		gnn_pred = prob2pred(gnn_prob_arr, thres)

		gnn_pred_list.extend(gnn_pred.tolist())
		gnn_prob_list.extend(gnn_prob_arr.tolist())

	auc_gnn = roc_auc_score(labels, np.array(gnn_prob_list))
	f1_macro_gnn = f1_score(labels, np.array(gnn_pred_list), average='macro')
	conf_gnn = confusion_matrix(labels, np.array(gnn_pred_list))
	tn, fp, fn, tp = conf_gnn.ravel()
	gmean_gnn = conf_gmean(conf_gnn)

	logging.info(f"\tF1-macro: {f1_macro_gnn:.4f}\tG-Mean: {gmean_gnn:.4f}\tAUC: {auc_gnn:.4f}")
	logging.info(f"   GNN TP: {tp}\tTN: {tn}\tFN: {fn}\tFP: {fp}")
	return f1_macro_gnn, auc_gnn, gmean_gnn

def conf_gmean(conf):
	tn, fp, fn, tp = conf.ravel()
	return (tp*tn/((tp+fn)*(tn+fp)))**0.5