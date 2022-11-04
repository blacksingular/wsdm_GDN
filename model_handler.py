import time, datetime
import os
import random
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

from utils.utils import test_GDN, test_sage, load_data, pos_neg_split, normalize, biased_split
from models.model import GDNLayer
from models.layers import InterAgg, IntraAgg
from models.graphsage import *
import pickle as pkl
import logging
import torch
import torch.nn as nn

timestamp = time.time()
timestamp = datetime.datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H-%M-%S')
logging.basicConfig(filename='result.log',level=logging.INFO)

"""
	Training GDN
"""


class ModelHandler(object):

	def __init__(self, config):
		args = argparse.Namespace(**config)
		# load graph, feature, and label
		[homo, relation1, relation2, relation3], feat_data, labels = load_data(args.data_name, prefix=args.data_dir)

		# train_test split
		np.random.seed(args.seed)
		random.seed(args.seed)
		
		if not args.biased_split:
			if args.data_name == 'yelp':
				index = list(range(len(labels)))
				idx_rest, idx_test, y_rest, y_test = train_test_split(index, labels, stratify=labels, train_size=args.train_ratio,
																		random_state=2, shuffle=True)
				idx_train, idx_valid, y_train, y_valid = train_test_split(idx_rest, y_rest, stratify=y_rest, test_size=args.test_ratio,
																		random_state=2, shuffle=True)
			elif args.data_name == 'amazon':  # amazon
				# 0-3304 are unlabeled nodes
				index = list(range(3305, len(labels)))
				idx_rest, idx_test, y_rest, y_test = train_test_split(index, labels[3305:], stratify=labels[3305:],
																		train_size=args.train_ratio, random_state=2, shuffle=True)
				idx_train, idx_valid, y_train, y_valid = train_test_split(idx_rest, y_rest, stratify=y_rest, test_size=args.test_ratio,
																		random_state=2, shuffle=True)
		else:
			idx_rest, idx_test, y_rest, y_test = biased_split(args.data_name)
			idx_train, idx_valid, y_train, y_valid = train_test_split(idx_rest, y_rest, stratify=y_rest, test_size=args.test_ratio,
																		random_state=2, shuffle=True)

		print(f'Run on {args.data_name}, postive/total num: {np.sum(labels)}/{len(labels)}, train num {len(y_train)},'+
			f'valid num {len(y_valid)}, test num {len(y_test)}, test positive num {np.sum(y_test)}')
		print(f"Classification threshold: {args.thres}")
		print(f"Feature dimension: {feat_data.shape[1]}")


		# split pos neg sets for under-sampling
		train_pos, train_neg = pos_neg_split(idx_train, y_train)

		
		if args.data_name == 'amazon':
			feat_data = normalize(feat_data)

		args.cuda = not args.no_cuda and torch.cuda.is_available()
		os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id

		# set input graph
		if args.model == 'SAGE' or args.model == 'GCN':
			adj_lists = homo
		else:
			adj_lists = [relation1, relation2, relation3]

		print(f'Model: {args.model}, multi-relation aggregator: {args.multi_relation}, emb_size: {args.emb_size}.')
		
		self.args = args
		self.dataset = {'feat_data': feat_data, 'labels': labels, 'adj_lists': adj_lists, 'homo': homo,
						'idx_train': idx_train, 'idx_valid': idx_valid, 'idx_test': idx_test,
						'y_train': y_train, 'y_valid': y_valid, 'y_test': y_test,
						'train_pos': train_pos, 'train_neg': train_neg}


	def train(self):
		args = self.args
		feat_data, adj_lists = self.dataset['feat_data'], self.dataset['adj_lists']
		idx_train, y_train = self.dataset['idx_train'], self.dataset['y_train']
		idx_valid, y_valid, idx_test, y_test = self.dataset['idx_valid'], self.dataset['y_valid'], self.dataset['idx_test'], self.dataset['y_test']
		# initialize model input
		features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
		features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=True)
		if args.cuda:
			features.cuda()

		# build one-layer models
		if args.model == 'GDN':
			intra1 = IntraAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], cuda=args.cuda)
			intra2 = IntraAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], cuda=args.cuda)
			intra3 = IntraAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], cuda=args.cuda)
			inter1 = InterAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], self.dataset['train_neg'],
							  adj_lists, [intra1, intra2, intra3], inter=args.multi_relation, cuda=args.cuda)
		elif args.model == 'SAGE':
			agg_sage = MeanAggregator(features, cuda=args.cuda)
			enc_sage = Encoder(features, feat_data.shape[1], args.emb_size, adj_lists, agg_sage, self.dataset['train_pos'], self.dataset['train_neg'], gcn=False, cuda=args.cuda)
		elif args.model == 'GCN':
			agg_gcn = GCNAggregator(features, cuda=args.cuda)
			enc_gcn = GCNEncoder(features, feat_data.shape[1], args.emb_size, adj_lists, agg_gcn, self.dataset['train_pos'],
			self.dataset['train_neg'], gcn=True, cuda=args.cuda)

		if args.model == 'GDN':
			gnn_model = GDNLayer(2, inter1)
		elif args.model == 'SAGE':
			# the vanilla GraphSAGE model as baseline
			enc_sage.num_samples = 5
			gnn_model = GraphSage(2, enc_sage)
		elif args.model == 'GCN':
			gnn_model = GCN(2, enc_gcn)

		if args.cuda:
			gnn_model.cuda()

		if args.model == 'GDN' or 'SAGE' or 'GCN':
			group_1 = []
			group_2 = []
			for name, param in gnn_model.named_parameters():
				print(name)
				if name == 'inter1.features.weight':
					group_2 += [param]
				else:
					group_1 += [param]
			optimizer = torch.optim.Adam([
				dict(params=group_1, weight_decay=args.weight_decay, lr=args.lr_1),
				dict(params=group_2, weight_decay=args.weight_decay_2, lr=args.lr_2)
				], )
		else:
			optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gnn_model.parameters()), lr=args.lr_1, weight_decay=args.weight_decay)

		dir_saver = args.save_dir+timestamp
		path_saver = os.path.join(dir_saver, '{}_{}.pkl'.format(args.data_name, args.model))
		f1_mac_best, auc_best, ep_best = 0, 0, -1

		# train the model
		for epoch in range(args.num_epochs):
			num_batches = int(len(idx_train) / args.batch_size) + 1

			loss = 0.0
			epoch_time = 0

			# mini-batch training
			for batch in range(num_batches):
				start_time = time.time()
				i_start = batch * args.batch_size
				i_end = min((batch + 1) * args.batch_size, len(idx_train))
				batch_nodes = idx_train[i_start:i_end]
				batch_label = self.dataset['labels'][np.array(batch_nodes)]
				optimizer.zero_grad()
				if args.cuda:
					loss = gnn_model.loss(batch_nodes, Variable(torch.cuda.LongTensor(batch_label)))
				else:
					loss = gnn_model.loss(batch_nodes, Variable(torch.LongTensor(batch_label)))
				loss.backward(retain_graph=True)

				# calculate grad and rank
				if args.add_constraint:
					# fl step
					if args.model == 'GDN':
						grad = torch.abs(torch.autograd.grad(outputs=loss, inputs=gnn_model.inter1.features.weight)[0])
					elif args.model == 'GCN' or 'SAGE':
						grad = torch.abs(torch.autograd.grad(outputs=loss, inputs=gnn_model.enc.features.weight)[0])
					grads_idx = grad.mean(dim=0).topk(k=args.topk).indices

					# find non grad idx
					mask_len = feat_data.shape[1] - args.topk
					non_grads_idx = torch.zeros(mask_len, dtype=torch.long)
					idx = 0
					for i in range(feat_data.shape[1]):
						if i not in grads_idx:
							non_grads_idx[idx] = i
							idx += 1
					
					if args.model == 'GDN':
						loss_pos, loss_neg = gnn_model.inter1.fl_loss(grads_idx)
					elif args.model == 'GCN' or 'SAGE':
						loss_pos, loss_neg = gnn_model.enc.constraint_loss(grads_idx)
					loss_stable = args.Beta * torch.exp((loss_pos - loss_neg))
					loss_stable.backward(retain_graph=True)

					# fn step
					if args.model == 'GDN':
						fn_pos, fn_neg = gnn_model.inter1.fn_loss(batch_nodes, non_grads_idx)
					elif args.model == 'GCN' or 'SAGE':
						fn_pos, fn_neg = gnn_model.enc.fn_loss(batch_nodes, non_grads_idx)
					loss_fn = args.Beta * torch.exp((fn_pos - fn_neg))
					loss_fn.backward()

				optimizer.step()
				end_time = time.time()
				epoch_time += end_time - start_time
				loss += loss.item()

			print(f'Epoch: {epoch}, loss: {loss.item()  / num_batches}, time: {epoch_time}s')

			# Valid the model for every $valid_epoch$ epoch
			if epoch % args.valid_epochs == 0:
				if args.model == 'SAGE' or args.model == 'GCN':
					print("Valid at epoch {}".format(epoch))
					f1_mac_val, auc_val, gmean_val = test_sage(idx_valid, y_valid, gnn_model, args.test_batch_size, args.thres)
					if auc_val > auc_best:
						auc_best, ep_best = auc_val, epoch
						if not os.path.exists(dir_saver):
							os.makedirs(dir_saver)
						print('  Saving model ...')
						torch.save(gnn_model.state_dict(), path_saver)
				else:
					print("Valid at epoch {}".format(epoch))
					f1_mac_val, auc_val, gmean_val = test_GDN(idx_valid, y_valid, gnn_model, args.batch_size, args.thres)
					if auc_val > auc_best:
						auc_best, ep_best = auc_val, epoch
						if not os.path.exists(dir_saver):
							os.makedirs(dir_saver)
						print('  Saving model ...')
						torch.save(gnn_model.state_dict(), path_saver)
						with open(args.data_name+'_features.pkl', 'wb+') as f:
							pkl.dump(gnn_model.inter1.features.weight, f)
	
		print("Restore model from epoch {}".format(ep_best))
		print("Model path: {}".format(path_saver))
		gnn_model.load_state_dict(torch.load(path_saver))

		if args.model == 'SAGE' or args.model == 'GCN':
			f1_mac_test, auc_test, gmean_test = test_sage(idx_test, y_test, gnn_model, args.test_batch_size, args.thres)
		else:
			f1_mac_test, auc_test, gmean_test = test_GDN(idx_test, y_test, gnn_model, args.batch_size, args.thres, True)
		return f1_mac_test, auc_test, gmean_test
