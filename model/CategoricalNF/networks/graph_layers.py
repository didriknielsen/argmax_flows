import torch
import torch.nn as nn
import numpy as np
import time
import sys
sys.path.append("../../")

from general.mutils import one_hot, get_device


#####################
## Node-based GNNs ##
#####################

class RelationGraphConv(nn.Module):

	def __init__(self, c_in, c_out, num_edges, **kwargs):
		super().__init__()
		self.c_in = c_in
		self.c_out = c_out
		self.num_edges = num_edges

		self.norm_layer = nn.LayerNorm(self.c_in)
		self.linear_hs = nn.Linear(self.c_in, self.c_out)
		self.linear_hr = nn.Linear(self.c_in, self.c_out*self.num_edges)

	def forward(self, x, adjacency, num_neighbours=None):
		"""
		Inputs:
			x - Node features. Shape: [batch, #Nodes, c_in]
			adjacency - One-hot adjacency matrix. Shape: [batch, #Nodes, #Nodes, num_edges] 
			num_neighbours - Count of neighbours for each node. Shape: [batch, #Nodes]
		"""
		batch_size, num_nodes = x.size(0), x.size(1)
		if num_neighbours is None:
			num_neighbours = adjacency.sum(dim=[1,3])
		
		x = self.norm_layer(x)
		hs = self.linear_hs(x)

		hr_all = self.linear_hr(x)
		hr_all =  hr_all.view( batch_size, num_nodes, 1,         self.num_edges, self.c_out)
		adjacency = adjacency.view(batch_size, num_nodes, num_nodes, self.num_edges, 1         )

		hr_adj = (hr_all * adjacency).sum(dim=[1,3]) # Shape: [batch_size, num_nodes, c_out]
		hr     = hr_adj / num_neighbours.unsqueeze(dim=-1).clamp(min=1e-5)

		h_out = hs + hr

		return h_out


class RelationGraphAttention(nn.Module):

	def __init__(self, c_in, c_out, num_edges, num_heads=4, **kwargs):
		super().__init__()
		self.c_in = c_in 
		self.c_out = c_out 
		self.num_edges = num_edges
		self.num_heads = num_heads
		self.c_out_per_head = self.c_out * 2 // self.num_heads

		self.norm_layer = nn.LayerNorm(self.c_in)
		self.linear_hs = nn.Linear(self.c_in, self.c_out_per_head * self.num_heads)
		self.linear_hr = nn.Linear(self.c_in, self.c_out_per_head * self.num_heads * (self.num_edges+1))
		self.attn_weight = nn.Parameter(torch.zeros(self.num_heads, 2, self.c_out_per_head), requires_grad=True)
		nn.init.xavier_uniform_(self.attn_weight.data, gain=1.414)
		self.output_projection = nn.Sequential(
				nn.GELU(),
				nn.Linear(self.c_out_per_head * self.num_heads, self.c_out)
			)

		self.leaky_relu = nn.LeakyReLU(0.2)


	def forward(self, x, adjacency, **kwargs):
		"""
		The forward-pass implementation of the layer is optimized for sparse graphs.
		Instead calculating the attention score for every pair of nodes and masking
		the scores afterwards, we first create a matrix of size [#nodes, #max_neighbours].
		The row [i] represents the features for (at least) all neighbours of the node i,
		plus additional padding. Calculating the attention on this matrix is especially
		efficient if #max_neighbours << graph_size.
		"""
		batch_size, num_nodes = x.size(0), x.size(1)

		## Run layers
		x = self.norm_layer(x)
		hs = self.linear_hs(x)
		hr_all = self.linear_hr(x)

		hs = hs.reshape(batch_size, num_nodes, self.num_heads, self.c_out_per_head)
		hs_attn = (hs * self.attn_weight[:,0].view(1, 1, self.num_heads, self.c_out_per_head)).sum(dim=-1)
		hr_all = hr_all.reshape(batch_size, num_nodes, self.num_edges+1, self.num_heads, self.c_out_per_head)
		hr_attn = (hr_all * self.attn_weight[:,1].view(1, 1, 1, self.num_heads, self.c_out_per_head)).sum(dim=-1)

		## Add self-connections
		with torch.no_grad():
			self_connections = torch.eye(num_nodes, device=x.device, dtype=adjacency.dtype)
			self_connections = self_connections.view(1, num_nodes, num_nodes, 1).repeat(batch_size, 1, 1, 1)
			adjacency = torch.cat([adjacency, self_connections], dim=-1)
			flat_adjacency = adjacency.sum(dim=-1)

			num_neighbours = flat_adjacency.sum(dim=2).long()
			max_neighbours = num_neighbours.max().item()

			# We need to add max_neighbours - 1 padding as because of self connections, we know each node has at least 1 connection
			padding_flat_adjacency = (torch.arange(1, max_neighbours, device=x.device).view(1, 1, -1) >= num_neighbours.view(batch_size, num_nodes, 1)).to(flat_adjacency.dtype)
			expanded_flat_adjacency = torch.cat([flat_adjacency, padding_flat_adjacency], dim=2)
			expanded_adjacency = torch.cat([adjacency, adjacency.new_zeros(batch_size, num_nodes, max_neighbours-1, adjacency.shape[-1])], dim=2)

			expanded_num_nodes = expanded_flat_adjacency.shape[2]
			num_elements = expanded_num_nodes * num_nodes
			flat_range = torch.arange(num_elements, device=x.device, dtype=torch.long)
			offset_range = num_elements * torch.arange(batch_size, device=x.device, dtype=flat_range.dtype)
			expanded_flat_range = flat_range[None,:] + offset_range[:,None]
			flat_indices = torch.masked_select(expanded_flat_range, expanded_flat_adjacency.view(batch_size, -1)==1.0).view(-1)

			def reduce_adjacency(tensor):
				tensor_reduced = tensor.reshape((batch_size * num_nodes * expanded_num_nodes,) + tensor.shape[3:]).index_select(index=flat_indices, dim=0)
				tensor_reduced = tensor_reduced.reshape((batch_size, num_nodes, max_neighbours) + tensor.shape[3:])
				return tensor_reduced
			adjacency_reduced = reduce_adjacency(expanded_adjacency)
			flat_adjacency_reduced = adjacency_reduced.sum(dim=-1)

			hr_flat_range = torch.arange(expanded_num_nodes, device=x.device, dtype=flat_range.dtype).repeat(num_nodes)
			hr_offset_range = expanded_num_nodes * torch.arange(batch_size, device=x.device, dtype=flat_range.dtype)
			exp_hr_flat_range = hr_flat_range[None,:] + hr_offset_range[:,None]
			flat_hr_indices = torch.masked_select(exp_hr_flat_range, expanded_flat_adjacency.view(batch_size, -1)==1.0).view(-1)

		def reduce_hr(tensor):
			tensor = torch.cat([tensor, tensor.new_zeros((batch_size, max_neighbours-1) + tensor.shape[2:])], dim=1)
			tensor_reduced = tensor.view((batch_size * expanded_num_nodes,)+tensor.shape[2:]).index_select(index=flat_hr_indices, dim=0)
			tensor_reduced = tensor_reduced.reshape((batch_size, num_nodes, max_neighbours) + tensor.shape[2:])
			return tensor_reduced

		hr_reduced = reduce_hr(hr_all)
		hr_attn_reduced = reduce_hr(hr_attn)

		hr = (hr_reduced * adjacency_reduced[...,None,None]).sum(dim=3)
		hr_attn = (hr_attn_reduced * adjacency_reduced[...,None]).sum(dim=3)

		## Determine attention weights
		attn_logits = hs_attn.unsqueeze(dim=2) + hr_attn
		attn_logits = self.leaky_relu(attn_logits)
		attn_logits = attn_logits.masked_fill(flat_adjacency_reduced.unsqueeze(dim=-1) == 0, -9e15)
		
		## Calculate output
		attn_probs = torch.softmax(attn_logits, dim=2)
		attn_output = (attn_probs.unsqueeze(dim=-1) * hr).sum(dim=2)
		attn_output = attn_output.reshape(batch_size, num_nodes, self.c_out_per_head * self.num_heads)
		h_out = self.output_projection(attn_output)

		return h_out


class RGCNNet(nn.Module):

	def __init__(self, c_in, c_out, num_edges, num_layers, hidden_size, dp_rate=0.0, max_neighbours=4, 
					   skip_config=2, rgc_layer_fun=RelationGraphConv, **kwargs):
		super().__init__()
		self.c_in = c_in
		self.c_out = c_out
		self.num_edges = num_edges
		self.num_layers = num_layers
		self.hidden_size = hidden_size
		self.dp_rate = dp_rate
		self.max_neighbours = max_neighbours

		if self.max_neighbours > 0:
			neighbour_embed_size = int(hidden_size // 4)
			self.neighbour_embed = nn.Linear(max_neighbours+1, neighbour_embed_size)
		else:
			neighbour_embed_size = 0

		self.act_fn = nn.GELU()
		self.dropout = nn.Dropout(dp_rate)
		self.layers = []
		for i in range(num_layers):
			self.layers += [
				nn.ModuleList([
					rgc_layer_fun(c_in=hidden_size,
								  c_out=hidden_size,
								  num_edges=num_edges),
					self.act_fn,
					self.dropout,
					GNNSkipConnection(hidden_size=hidden_size, config=skip_config)
				])
			]
		self.layers = nn.ModuleList(self.layers)

		self.input_layer = nn.Sequential(
				nn.Linear(c_in, hidden_size),
				self.act_fn,
				nn.Linear(hidden_size, hidden_size - neighbour_embed_size)
			)
		self.output_layer = nn.Sequential(
				nn.LayerNorm(hidden_size),
				nn.Linear(hidden_size, hidden_size),
				self.act_fn,
				nn.Linear(hidden_size, c_out)
			)

	def forward(self, x, adjacency, channel_padding_mask=None, embed_ext_input=None, **kwargs):
		adj_one_hot = one_hot(adjacency, num_classes=self.num_edges+1)[...,1:] # Remove zeros as those are no edges
		num_neighbours = adj_one_hot.sum(dim=[1,3])

		x_feat = self.input_layer(x)
		if self.max_neighbours > 0:
			num_neighbours = num_neighbours.clamp(max=self.max_neighbours)
			neigh_feat = self.neighbour_embed(one_hot(num_neighbours.long(), num_classes=self.max_neighbours+1))
			x = torch.cat([x_feat, neigh_feat], dim=-1)
		else:
			x = x_feat

		for layer_block in self.layers:
			x_orig = x 
			for layer in layer_block:
				if isinstance(layer, RelationGraphConv) or isinstance(layer, RelationGraphAttention):
					x = layer(x, adjacency=adj_one_hot, num_neighbours=num_neighbours)
				elif isinstance(layer, GNNSkipConnection):
					x = layer(orig=x_orig, feat=x)
				else: # Dropout and Activation
					x = layer(x)

		x = self.output_layer(x)

		# Channel padding mask only needs to be applied at last step because adjacency matrix
		# is zero for those elements anyways
		if channel_padding_mask is not None:
			x = x * channel_padding_mask
		return x




##############
## Edge GNN ##
##############


class EdgeGNNLayer(nn.Module):

	def __init__(self, edge2node_layer_func, node2edge_layer_func):
		super().__init__()
		self.node2edge_layer = node2edge_layer_func()
		self.edge2node_layer = edge2node_layer_func()

	def forward(self, node_feat, edge_feat, x_indices, mask_valid, **kwargs):
		node_feat = self.edge2node_layer(node_feat=node_feat,
										 edge_feat=edge_feat,
										 x_indices=x_indices,
										 mask_valid=mask_valid,
										 **kwargs)
		edge_feat = self.node2edge_layer(node_feat=node_feat,
										 edge_feat=edge_feat,
										 x_indices=x_indices,
										 mask_valid=mask_valid,
										 **kwargs)

		edge_feat = edge_feat * mask_valid.unsqueeze(dim=-1)
		return node_feat, edge_feat

	@staticmethod
	def _sort_edges_by_nodes(edge_feat, x_indices, num_nodes, sort_indices=None, edge_feat_list=None, mask_valid=None):
		# Strategy: each node has (node minus 1) edges. Order the edge feature array to [node, node-1] and sum second dimension
		# 1.) Step: find sorting indices to have edge pairs like [n1_n2, n1_n3, ..., n2_n1, n2_n3, ..., n3_n1, n3_n2, n3_n4, ...]
		if sort_indices is None:
			sort_indices = EdgeGNNLayer._get_sort_indices(x_indices)
		# 2.) Step: apply sorting and reshape
		if edge_feat_list is None:
			edge_feat_list = [edge_feat, edge_feat]
		edge_feat = torch.cat(edge_feat_list, dim=1).index_select(index=sort_indices, dim=1)
		edge_feat = edge_feat.reshape(edge_feat.size(0), num_nodes, num_nodes-1, edge_feat.size(-1))
		if mask_valid is None:
			return edge_feat
		else:
			mask_valid = torch.cat([mask_valid, mask_valid], dim=1).index_select(index=sort_indices, dim=1)
			mask_valid = mask_valid.reshape(mask_valid.size(0), num_nodes, num_nodes-1)
			return edge_feat, mask_valid

	@staticmethod
	def _get_sort_indices(x_indices):
		x_indices1, x_indices2 = x_indices
		x_indices = torch.cat([x_indices1, x_indices2], dim=0) # Shape: [#pairs*2]
		_, sort_indices = x_indices.sort(0, descending=False)
		return sort_indices

	@staticmethod
	def _get_node_feat_by_indices(node_feat, x_indices, dim=1):
		x_indices1, x_indices2 = x_indices
		node1_feat = node_feat.index_select(index=x_indices1, dim=dim)
		node2_feat = node_feat.index_select(index=x_indices2, dim=dim)
		return node1_feat, node2_feat


class Node2EdgePlainLayer(nn.Module):
	"""
	Simple edge update layer, where for each edge we consider the adjacent node features.
	"""

	def __init__(self, hidden_size_nodes, hidden_size_edges, skip_config=0, dp_rate=0.0, act_fn=nn.GELU):
		super().__init__()
		self.hidden_size_nodes = hidden_size_nodes
		self.hidden_size_edges = hidden_size_edges

		self.node_feat_layer = nn.Linear(hidden_size_nodes, hidden_size_edges)
		self.edge_feat_layer = nn.Linear(hidden_size_edges, hidden_size_edges)
		self.skip_layer = GNNSkipConnection(hidden_size_edges, config=skip_config,
											dp_rate=dp_rate)
		self.dropout = nn.Dropout(dp_rate)
		self.act_fn = act_fn()

		self.node_feat_layer = nn.Sequential(nn.LayerNorm(hidden_size_nodes), self.node_feat_layer)
		self.edge_feat_layer = nn.Sequential(nn.LayerNorm(hidden_size_edges), self.edge_feat_layer)

	def forward(self, node_feat, edge_feat, x_indices, mask_valid, 
					  flat_indices=None, indices_reverse=None, **kwargs):
		node_feat_linear = self.node_feat_layer(self.dropout(node_feat))
		# Sort the node features by edges
		node1_feat, node2_feat = EdgeGNNLayer._get_node_feat_by_indices(node_feat_linear, x_indices)
		node_sum_feat = node1_feat + node2_feat

		def module_func(edge_feat_in, node_sum_feat_in): 
			edge_feat_linear = self.edge_feat_layer(self.dropout(edge_feat_in))
			comb_feat = self.act_fn(self.dropout(edge_feat_linear + node_sum_feat_in))
			edge_feat_out = self.skip_layer(orig=edge_feat_in, feat=comb_feat)
			return edge_feat_out

		edge_feat = _run_module_effec(module_func, edge_feat, mask_valid, [node_sum_feat],
									  flat_indices=flat_indices, indices_reverse=indices_reverse)

		return edge_feat

#--------------------------------------------------#
#-- Functions for running edge layer efficiently --#
#--------------------------------------------------#

def _create_flat_indices(mask_valid):
	batch_size, seq_len = mask_valid.size(0), mask_valid.size(1)

	flat_range = torch.arange(seq_len, device=mask_valid.device, dtype=torch.long)
	offset_range = seq_len * torch.arange(batch_size, device=mask_valid.device, dtype=flat_range.dtype)
	expanded_flat_range = flat_range[None,:] + offset_range[:,None]
	flat_indices = torch.masked_select(expanded_flat_range, mask_valid.view(batch_size, -1)==1.0).view(-1)

	return flat_indices

def _create_reverse_indices(mask_valid):
	batch_size, seq_len = mask_valid.size(0), mask_valid.size(1)

	flat_mask = mask_valid.view(batch_size * seq_len).long()
	indices_reverse = flat_mask * (flat_mask).cumsum(dim=0)

	return indices_reverse

def _run_module_effec(module, x, mask_valid, add_args=None, flat_indices=None, indices_reverse=None, pad_val=0.0, **kwargs):
	global DURATION
	if add_args is None:
		add_args = []
	batch_size, seq_len = x.size(0), x.size(1)

	if flat_indices is None:
		flat_indices = _create_flat_indices(mask_valid)

	def reduce_tensor(tensor):
		tensor_reduced = tensor.reshape((batch_size * seq_len,) + tensor.shape[2:]).index_select(index=flat_indices, dim=0)
		return tensor_reduced

	x_reduced = reduce_tensor(x)
	add_args_reduced = [reduce_tensor(t) for t in add_args]

	x_out_reduced = module(x_reduced, *add_args_reduced)
	## Padding
	x_out_reduced = torch.cat([x_out_reduced.new_zeros((1,)+x_out_reduced.shape[1:])+pad_val, x_out_reduced], dim=0)

	if indices_reverse is None:
		indices_reverse = _create_reverse_indices(mask_valid)

	def reverse_tensor(tensor_reduced):
		tensor = tensor_reduced.index_select(index=indices_reverse, dim=0).reshape((batch_size, seq_len) + tensor_reduced.shape[1:])
		return tensor

	x_out = reverse_tensor(x_out_reduced)
	return x_out


class Edge2NodeQKVAttnLayer(nn.Module):
	"""
	Node update layer based on transformer architecture. See paper appendix for details.
	"""

	def __init__(self, hidden_size_nodes, hidden_size_edges, num_heads=4, dp_rate=0.0, act_fn=nn.GELU,
					   skip_config=2):
		super().__init__()
		self.hidden_size_nodes = hidden_size_nodes
		self.hidden_size_edges = hidden_size_edges
		self.num_heads = num_heads 
		self.hidden_size_per_head = self.hidden_size_nodes // self.num_heads
		self.dot_prod_scaling = float(self.hidden_size_per_head) ** -0.5

		self.node_query_key_val_layer = nn.Linear(hidden_size_nodes, self.num_heads * self.hidden_size_per_head * 3)
		self.edge_val_layer = nn.Linear(hidden_size_edges, self.num_heads * self.hidden_size_per_head)
		self.edge_adj_layer = nn.Linear(hidden_size_edges, self.num_heads)
		self.output_projection = nn.Linear(self.num_heads * self.hidden_size_per_head + self.hidden_size_nodes, self.hidden_size_nodes)

		self.skip_layer = GNNSkipConnection(hidden_size_nodes, config=skip_config, 
											input_size=self.hidden_size_nodes,
											dp_rate=dp_rate)

		self.dropout = nn.Dropout(dp_rate)
		self.act_fn = act_fn()
		self.node_normalization = nn.LayerNorm(hidden_size_nodes)
		self.edge_normalization = nn.LayerNorm(hidden_size_edges)


	def forward(self, *args, **kwargs):
		"""
		We implement two forward passes for efficiency. If a binary adjacency matrix is given (which is assumed to be sparse),
		we apply a similar approach as for GAT and align each node with its neighbours before calculating the attention scores. 
		This is especially helpful for molecules which have up to 38 nodes but always less than 5 neighbouring nodes.
		If this matrix is not given, we assume (from the efficiency perspective) a fully-connected graph and hence, calculate the 
		full attention matrix, with eventual masking afterwards.
		The output of both forward passes are exactly the same.
		"""
		if "binary_adjacency" in kwargs and kwargs["binary_adjacency"] is not None:
			return self.forward_sparse(*args, **kwargs)
		else:
			return self.forward_full(*args, **kwargs)


	def forward_full(self, node_feat, edge_feat, x_indices, mask_valid, sort_indices=None, 
					  channel_padding_mask=None, **kwargs):
		batch_size, num_nodes = node_feat.size(0), node_feat.size(1)
		num_pairs = edge_feat.size(1)

		## Run all necessary layers and obtain value, query and key vectors
		node_feat_in, edge_feat_in = self.node_normalization(node_feat), self.edge_normalization(edge_feat)
		node_query, node_key, node_val = self.node_query_key_val_layer(self.dropout(node_feat_in)).chunk(3, dim=-1)
		edge_val = self.edge_val_layer(edge_feat_in)
		edge_adj = self.edge_adj_layer(edge_feat_in)
		orig_node_val = node_val

		## Align all vectors along dimensions
		node_query = node_query.reshape(batch_size, num_nodes, 1, self.num_heads, self.hidden_size_per_head).permute(0, 3, 1, 2, 4).contiguous()
		node_key   =   node_key.reshape(batch_size, 1, num_nodes, self.num_heads, self.hidden_size_per_head).permute(0, 3, 1, 2, 4).contiguous()
		node_val   =   node_val.reshape(batch_size, num_nodes, self.num_heads, self.hidden_size_per_head).permute(0, 2, 1, 3).contiguous()
		edge_val   =   edge_val.reshape(batch_size, -1, self.num_heads, self.hidden_size_per_head).permute(0, 2, 1, 3).contiguous()
		edge_adj   = edge_adj.permute(0, 2, 1).contiguous()
		
		## Calculate the attention scores and integrate the adjustment factor by the edges
		attn_logits = (node_query * node_key).sum(dim=-1) * self.dot_prod_scaling
		attn_logits = attn_logits.view(batch_size * self.num_heads, -1)
		x_flat_indices1 = x_indices[0] + x_indices[1] * num_nodes
		x_flat_indices2 = x_indices[1] + x_indices[0] * num_nodes
		attn_logits_by_pairs1 = attn_logits.index_select(index=x_flat_indices1, dim=1).reshape(batch_size, self.num_heads, -1)
		attn_logits_by_pairs2 = attn_logits.index_select(index=x_flat_indices2, dim=1).reshape(batch_size, self.num_heads, -1)
		attn_logits_by_pairs1 = torch.stack([attn_logits_by_pairs1, edge_adj], dim=-1)
		attn_logits_by_pairs2 = torch.stack([attn_logits_by_pairs2, edge_adj], dim=-1)
		
		## Obtain node+edge features per node
		node1_val_feat, node2_val_feat = EdgeGNNLayer._get_node_feat_by_indices(node_val, x_indices, dim=2)
		edge_feat_list = [torch.cat([attn_logits_by_pairs2, (edge_val+node2_val_feat)], dim=-1), 
						  torch.cat([attn_logits_by_pairs1, (edge_val+node1_val_feat)], dim=-1)]
		edge_feat_list = [e.reshape(batch_size*self.num_heads, num_pairs, -1) for e in edge_feat_list]
		node_edge_log_val, mask_valid = EdgeGNNLayer._sort_edges_by_nodes(edge_feat=edge_feat_list[0], 
																			x_indices=x_indices, 
																			num_nodes=node_feat.size(1), 
																			sort_indices=sort_indices,
																			edge_feat_list=edge_feat_list,
																			mask_valid=mask_valid)
		node_edge_attn_logits, node_edge_val = node_edge_log_val[...,:2], node_edge_log_val[...,2:]

		## Calculate final attention probability matrix
		mask_valid = mask_valid[:,None,:,:,None].repeat(1,self.num_heads,1,1,1) # Shape: [batch, num_heads, node, node-1, 1]
		node_edge_val = node_edge_val.reshape(batch_size, self.num_heads, num_nodes, num_nodes-1, self.hidden_size_per_head)
		attn_logits = node_edge_attn_logits.reshape(batch_size, self.num_heads, num_nodes, num_nodes-1, 2)
		attn_logits = attn_logits.sum(dim=-1, keepdims=True)
		attn_logits = attn_logits.masked_fill(mask_valid==0.0, -9e15)
		attn_probs = torch.softmax(attn_logits, dim=3)
		attn_probs = attn_probs * (1 - torch.all(mask_valid==0.0, dim=3, keepdims=True).float())

		## Calculate final attention features and update node features
		attn_output = (node_edge_val * attn_probs).sum(dim=3)
		attn_output = attn_output.permute(0, 2, 1, 3)
		attn_output = attn_output.reshape(batch_size, num_nodes, self.num_heads * self.hidden_size_per_head)

		comb_feat = self.act_fn(self.dropout(self.output_projection(torch.cat([node_feat_in, attn_output], dim=-1))))
		node_feat = self.skip_layer(orig=node_feat, feat=comb_feat)

		assert torch.isnan(node_feat).sum() == 0, "[!] ERROR: Detected NaN values in Edge2NodeQKVAttnLayer.\n" + \
				"\n".join(["-> %s: %s [%s]" % (name, str(torch.isnan(tensor).sum().item()), str(tensor.view(-1).shape[0])) for name, tensor in \
				  [("Node feat", node_feat), ("Edge feat", edge_feat), ("Attention output", attn_output),
				   ("Attention logits", attn_logits), ("Attention probs", attn_probs),
				   ("Comb feat", comb_feat), ("Node feat in", node_feat_in), ("Edge feat in", edge_feat_in),
				   ("Node Edge Attn logits", node_edge_attn_logits), ("Node Edge val", node_edge_val),
				   ("Attention logits by pairs 1", attn_logits_by_pairs1), ("Attention logits by pairs 2", attn_logits_by_pairs2),
				   ("Edge val", edge_val), ("Edge adj", edge_adj)
				   ]])

		return node_feat


	def forward_sparse(self, node_feat, edge_feat, x_indices, mask_valid, sort_indices=None, 
					   channel_padding_mask=None, binary_adjacency=None, **kwargs):
		batch_size, num_nodes = node_feat.size(0), node_feat.size(1)
		num_pairs = edge_feat.size(1)
		## Determine the "k" neighbours for each node. If a node has less than "k" neighbours, we pad them with random values and mask them later
		max_neighbours = binary_adjacency.sum(dim=-1).max().item()
		adjacency_mask, node_indices = torch.topk(binary_adjacency, k=max_neighbours, dim=-1, sorted=True)
		
		## Calculate key, query and value vectors for nodes
		node_feat_in = self.node_normalization(node_feat)
		node_query, node_key, node_val = self.node_query_key_val_layer(self.dropout(node_feat_in)).chunk(3, dim=-1)

		## Align the key and value vectors in the matrix [#nodes, max_neighbours]
		node_batch_indices = (node_indices + num_nodes * torch.arange(batch_size, dtype=torch.long, device=node_indices.device)[:,None,None]).view(-1)
		node_key = node_key.view(-1, node_key.size(-1)).index_select(index=node_batch_indices, dim=0).reshape(node_indices.shape + node_key.shape[-1:])
		node_val = node_val.view(-1, node_val.size(-1)).index_select(index=node_batch_indices, dim=0).reshape(node_indices.shape + node_val.shape[-1:])

		## Align the edges accordingly as well
		edge_indices_1 = torch.arange(num_nodes, dtype=torch.long, device=node_feat.device)
		edge_indices_1 = edge_indices_1.repeat_interleave(max_neighbours, dim=0)[None,:].expand(batch_size, -1) # Shape: [Batch-size, #nodes*max_neighbours]
		edge_indices_2 = node_indices.view(batch_size, -1)
		edge_indices, _ = torch.stack([edge_indices_1, edge_indices_2], dim=-1).sort(dim=-1)
		# Simple function to convert the two node indices per edge back to an index in the edge list
		edge_indices = (num_nodes - 1 - edge_indices[...,0]) * edge_indices[...,0] + ((edge_indices[...,0] + 1) * edge_indices[...,0]) / 2 + (edge_indices[...,1] - edge_indices[...,0] - 1)
		edge_indices = edge_indices.clamp(min=0, max=edge_feat.shape[1]-1) # Clamp as padded values might result in indices out of bounds
		edge_batch_indices = (edge_indices + edge_feat.shape[1] * torch.arange(batch_size, dtype=torch.long, device=edge_indices.device)[:,None]).view(-1)
		edge_feat = edge_feat.view(-1, edge_feat.size(-1)).index_select(index=edge_batch_indices, dim=0).reshape(edge_indices.shape + edge_feat.shape[-1:])

		## Determine value and attention adjustment score for each edge. We apply the layers after aligning as many previous might have been invalid/masked
		edge_feat_in = self.edge_normalization(edge_feat)
		edge_val = self.edge_val_layer(edge_feat_in)
		edge_adj = self.edge_adj_layer(edge_feat_in)

		## Align tensors by dimensions
		node_query = node_query.reshape(batch_size, num_nodes, 1             , self.num_heads, self.hidden_size_per_head)
		node_key   =   node_key.reshape(batch_size, num_nodes, max_neighbours, self.num_heads, self.hidden_size_per_head)
		node_val   =   node_val.reshape(batch_size, num_nodes, max_neighbours, self.num_heads, self.hidden_size_per_head)
		edge_val   =   edge_val.reshape(batch_size, num_nodes, max_neighbours, self.num_heads, self.hidden_size_per_head)
		edge_adj   =   edge_adj.reshape(batch_size, num_nodes, max_neighbours, self.num_heads)
		
		## Calculate final attention probability matrix
		attn_logits = (node_query * node_key).sum(dim=-1) * self.dot_prod_scaling
		attn_logits = attn_logits + edge_adj
		attn_logits = attn_logits.masked_fill(adjacency_mask[...,None]==0.0, -9e15)
		attn_probs = torch.softmax(attn_logits, dim=2)
		attn_probs = attn_probs * (1 - torch.all(adjacency_mask[...,None]==0.0, dim=2, keepdims=True).float())

		## Calculate final attention features and update node features
		attn_output = ((node_val + edge_val) * attn_probs[...,None]).sum(dim=2)
		attn_output = attn_output.reshape(batch_size, num_nodes, self.num_heads * self.hidden_size_per_head)
		
		comb_feat = self.act_fn(self.dropout(self.output_projection(torch.cat([node_feat_in, attn_output], dim=-1))))
		node_feat = self.skip_layer(orig=node_feat, feat=comb_feat)

		return node_feat


class Edge2NodeAttnLayer(nn.Module):
	"""
	Node update layer with attention purely based on edges. See paper appendix for details.
	"""

	def __init__(self, hidden_size_nodes, hidden_size_edges, skip_config=2, num_heads=4, dp_rate=0.0, act_fn=nn.GELU):
		super().__init__()
		self.hidden_size_nodes = hidden_size_nodes
		self.hidden_size_edges = hidden_size_edges
		self.num_heads = num_heads 
		self.hidden_size_per_head = int(self.hidden_size_nodes // self.num_heads)
		self.hidden_size_output = self.hidden_size_per_head * self.num_heads

		self.node_feat_layer = nn.Linear(hidden_size_nodes, self.hidden_size_output * 2)
		self.edge_feat_layer = nn.Linear(hidden_size_edges, self.hidden_size_output)
		self.edge_logits_layer = nn.Linear(hidden_size_edges, self.num_heads)
		self.skip_layer = GNNSkipConnection(hidden_size_nodes, config=skip_config, input_size=self.hidden_size_output)

		self.dropout = nn.Dropout(dp_rate)
		self.act_fn = act_fn()

		self.node_normalization = nn.LayerNorm(hidden_size_nodes)
		self.edge_normalization = nn.LayerNorm(hidden_size_edges)


	def forward(self, *args, **kwargs):
		"""
		We implement two forward passes for efficiency. See Edge2NodeQKVAttnLayer for details.
		"""
		if "binary_adjacency" in kwargs and kwargs["binary_adjacency"] is not None:
			return self.forward_sparse(*args, **kwargs)
		else:
			return self.forward_full(*args, **kwargs)

	def forward_full(self, node_feat, edge_feat, x_indices, mask_valid, sort_indices=None,
					  flat_indices=None, indices_reverse=None, **kwargs):
		batch_size, num_nodes = node_feat.size(0), node_feat.size(1)

		## Run all necessary layers
		node_feat_inp = self.node_normalization(node_feat)
		node_new_feat = self.node_feat_layer(node_feat_inp)
		node_self_feat, node_context_feat = node_new_feat.chunk(2, dim=-1)

		def module_func(edge_feat_in): 
			edge_feat_in = self.edge_normalization(edge_feat_in)
			edge_new_feat = self.edge_feat_layer(edge_feat_in)
			edge_logits = self.edge_logits_layer(edge_feat_in)
			return torch.cat([edge_logits, edge_new_feat], dim=-1)

		edge_out_feat = _run_module_effec(module_func, edge_feat, mask_valid,
										  flat_indices=flat_indices, indices_reverse=indices_reverse)
		edge_new_feat, edge_logits = edge_out_feat[...,self.num_heads:], edge_out_feat[...,:self.num_heads]
		
		## Obtain node-edge feature pairs
		node1_feat, node2_feat = EdgeGNNLayer._get_node_feat_by_indices(node_context_feat, x_indices)
		edge_feat_list = [edge_new_feat+node2_feat, edge_new_feat+node1_feat]
		node_edge_pair_feat, mask_valid = EdgeGNNLayer._sort_edges_by_nodes(edge_feat=edge_new_feat, 
																			x_indices=x_indices, 
																			num_nodes=num_nodes, 
																			sort_indices=sort_indices,
																			edge_feat_list=edge_feat_list,
																			mask_valid=mask_valid)
		edge_logits_list = [edge_logits, edge_logits]
		node_edge_logits = EdgeGNNLayer._sort_edges_by_nodes(edge_feat=edge_new_feat, 
																			x_indices=x_indices, 
																			num_nodes=num_nodes, 
																			sort_indices=sort_indices,
																			edge_feat_list=edge_logits_list,
																			mask_valid=None)
		mask_valid = mask_valid[:,:,:,None].repeat(1,1,1,self.num_heads) # Shape: [batch, node, node-1, num_heads]
		node_edge_pair_feat = node_edge_pair_feat.reshape(batch_size, num_nodes, num_nodes-1, self.num_heads, self.hidden_size_per_head)
		node_edge_logits = node_edge_logits.reshape(batch_size, num_nodes, num_nodes-1, self.num_heads)

		attn_logits = torch.where(mask_valid > 0.0, node_edge_logits, -9e15*torch.ones_like(mask_valid)) # Masking logits
		attn_sigmoid = torch.sigmoid(attn_logits)
		attn_probs = attn_sigmoid / attn_sigmoid.sum(dim=-2, keepdims=True).clamp(min=1e-5)

		attn_output = (node_edge_pair_feat * attn_probs.unsqueeze(dim=-1)).sum(dim=2)
		attn_output = attn_output.reshape(batch_size, num_nodes, self.hidden_size_output)

		comb_feat = self.act_fn(self.dropout(node_self_feat + attn_output))
		node_feat = self.skip_layer(orig=node_feat, feat=comb_feat)
		
		return node_feat


	def forward_sparse(self, node_feat, edge_feat, x_indices, mask_valid, binary_adjacency=None, sort_indices=None, **kwargs):
		batch_size, num_nodes = node_feat.size(0), node_feat.size(1)

		max_neighbours = binary_adjacency.sum(dim=-1).max().item()
		adjacency_mask, node_indices = torch.topk(binary_adjacency, k=max_neighbours, dim=-1, sorted=True)
		
		## Calculate self-connected values and vectors for nodes
		node_feat_inp = self.node_normalization(node_feat)
		node_new_feat = self.node_feat_layer(node_feat_inp)
		node_self_feat, node_context_feat = node_new_feat.chunk(2, dim=-1)

		## Align the key and value vectors in the matrix [#nodes, max_neighbours]
		node_batch_indices = (node_indices + num_nodes * torch.arange(batch_size, dtype=torch.long, device=node_indices.device)[:,None,None]).view(-1)
		node_context_feat = node_context_feat.view(-1, node_context_feat.size(-1)).index_select(index=node_batch_indices, dim=0).reshape(node_indices.shape + node_context_feat.shape[-1:])

		## Align the edges accordingly as well
		edge_indices_1 = torch.arange(num_nodes, dtype=torch.long, device=node_feat.device)
		edge_indices_1 = edge_indices_1.repeat_interleave(max_neighbours, dim=0)[None,:].expand(batch_size, -1) # Shape: [Batch-size, #nodes*max_neighbours]
		edge_indices_2 = node_indices.view(batch_size, -1)
		edge_indices, _ = torch.stack([edge_indices_1, edge_indices_2], dim=-1).sort(dim=-1)
		# Simple function to convert the two node indices per edge back to an index in the edge list
		edge_indices = (num_nodes - 1 - edge_indices[...,0]) * edge_indices[...,0] + ((edge_indices[...,0] + 1) * edge_indices[...,0]) / 2 + (edge_indices[...,1] - edge_indices[...,0] - 1)
		edge_indices = edge_indices.clamp(min=0, max=edge_feat.shape[1]-1) # Clamp as padded values might result in indices out of bounds
		edge_batch_indices = (edge_indices + edge_feat.shape[1] * torch.arange(batch_size, dtype=torch.long, device=edge_indices.device)[:,None]).view(-1)
		edge_feat = edge_feat.view(-1, edge_feat.size(-1)).index_select(index=edge_batch_indices, dim=0).reshape(edge_indices.shape + edge_feat.shape[-1:])

		## Determine value and attention score for each edge. We apply the layers after aligning as many previous might have been invalid/masked
		edge_feat_in = self.edge_normalization(edge_feat)
		edge_new_feat = self.edge_feat_layer(edge_feat_in)
		edge_logits = self.edge_logits_layer(edge_feat_in)

		## Align tensors by dimensions
		node_context_feat = node_context_feat.reshape(batch_size, num_nodes, max_neighbours, self.num_heads, self.hidden_size_per_head)
		edge_new_feat = edge_new_feat.reshape(batch_size, num_nodes, max_neighbours, self.num_heads, self.hidden_size_per_head)
		edge_logits = edge_logits.reshape(batch_size, num_nodes, max_neighbours, self.num_heads)
		
		## Calculate final attention probability matrix
		attn_logits = edge_logits
		attn_logits = attn_logits.masked_fill(adjacency_mask[...,None]==0.0, -9e15)
		# We apply a sigmoid-based attention here instead of a softmax as multiple edges might be equally important.
		attn_sigmoid = torch.sigmoid(attn_logits)
		attn_probs = attn_sigmoid / attn_sigmoid.sum(dim=2, keepdims=True).clamp(min=1e-5)
		attn_probs = attn_probs * (1 - torch.all(adjacency_mask[...,None]==0.0, dim=2, keepdims=True).float())

		## Calculate final attention features and update node features
		attn_output = ((edge_new_feat + node_context_feat) * attn_probs.unsqueeze(dim=-1)).sum(dim=2)
		attn_output = attn_output.reshape(batch_size, num_nodes, self.hidden_size_output)

		comb_feat = self.act_fn(self.dropout(node_self_feat + attn_output))
		node_feat = self.skip_layer(orig=node_feat, feat=comb_feat)
		
		return node_feat



class GNNSkipConnection(nn.Module):

	def __init__(self, hidden_size, config=0, input_size=-1, dp_rate=0.0):
		super().__init__()
		self.hidden_size = hidden_size
		self.input_size = input_size if input_size > 0 else hidden_size
		self.config = config 
		self.dp_rate = dp_rate

		if self.config == 0: # Config 0 => Simple connection with single feedforward
			self.skip_layer = nn.Linear(self.input_size, self.hidden_size)
		elif self.config == 1: # Config 2 => Gated skip connection
			self.skip_layer = nn.Linear(self.input_size, self.hidden_size*2)
		elif self.config == 2: # Config 3 => High-way network
			self.skip_layer = nn.Linear(self.input_size, self.hidden_size*2)
		else:
			assert False, "[!] ERROR: Unknown skip connection config \"%s\"" % str(self.config)
		if self.dp_rate > 0.0 and hasattr(self, "skip_layer"):
			self.skip_layer = nn.Sequential(nn.Dropout(self.dp_rate), self.skip_layer)

	def forward(self, orig, feat):
		if self.config == 0:
			out = orig + self.skip_layer(feat)
		elif self.config == 1:
			val, gate_logits = self.skip_layer(feat).chunk(2, dim=-1)
			gate_probs = torch.sigmoid(gate_logits)
			out = orig + val * gate_probs
		elif self.config == 2:
			val, gate_logits = self.skip_layer(feat).chunk(2, dim=-1)
			gate_probs = torch.sigmoid(gate_logits)
			out = orig * (1 - gate_probs) + val * gate_probs
		return out



class EdgeGNN(nn.Module):
	"""
	Overall model class that combines the Edge-GNN layers with input and output layers for flow transformations.
	"""

	def __init__(self, c_in_nodes, c_in_edges, c_out_nodes, c_out_edges, edge_gnn_layer_func, 
					   num_layers=4, max_neighbours=-1):
		super().__init__()
		self.c_in_nodes = c_in_nodes
		self.c_in_edges = c_in_edges
		self.c_out_nodes = c_out_nodes
		self.c_out_edges = c_out_edges

		self.layers = nn.ModuleList([
				edge_gnn_layer_func() for _ in range(num_layers)
			])

		hidden_size_edges = self.layers[0].node2edge_layer.hidden_size_edges
		hidden_size_nodes = self.layers[0].node2edge_layer.hidden_size_nodes
		
		self.input_layer_edges = self._create_input_network(c_in_edges, hidden_size_edges)
		self.input_layer_nodes = self._create_input_network(c_in_nodes, hidden_size_nodes)
		self.out_layer_edges = self._create_output_network(hidden_size_edges, c_out_edges)
		self.out_layer_nodes = self._create_output_network(hidden_size_nodes, c_out_nodes)

		if max_neighbours > 0:
			self.max_neighbours = max_neighbours
			self.node_neighbour_embed = nn.Linear(max_neighbours+1, hidden_size_nodes)

	def _create_input_network(self, c_in, hidden_size):
		return nn.Sequential(
				nn.Linear(c_in, hidden_size),
				nn.GELU(),
				nn.Linear(hidden_size, hidden_size)
			)

	def _create_output_network(self, hidden_size, c_out):
		return nn.Sequential(
				nn.LayerNorm(hidden_size),
				nn.Linear(hidden_size, hidden_size),
				nn.GELU(),
				nn.Linear(hidden_size, c_out)
			)

	def forward(self, z_nodes, z_edges, length, x_indices, mask_valid, channel_padding_mask=None,
					  binary_adjacency=None, **kwargs):
		nodes_feat = self.input_layer_nodes(z_nodes)
		edges_feat = self.input_layer_edges(z_edges)
		if binary_adjacency is not None and hasattr(self, "node_neighbour_embed"):
			num_neighbours = binary_adjacency.sum(dim=-1).long().clamp(max=self.max_neighbours)
			nodes_feat = nodes_feat + self.node_neighbour_embed(one_hot(num_neighbours, self.max_neighbours+1))

		sort_indices = EdgeGNNLayer._get_sort_indices(x_indices)
		flat_indices = _create_flat_indices(mask_valid)
		indices_reverse = _create_reverse_indices(mask_valid)
		for layer in self.layers:
			nodes_feat, edges_feat = layer(node_feat=nodes_feat,
										   edge_feat=edges_feat,
										   x_indices=x_indices,
										   sort_indices=sort_indices,
										   mask_valid=mask_valid,
										   flat_indices=flat_indices,
										   indices_reverse=indices_reverse,
										   channel_padding_mask=channel_padding_mask,
										   binary_adjacency=binary_adjacency,
										   **kwargs)

		nodes_out = self.out_layer_nodes(nodes_feat)
		edges_out = _run_module_effec(self.out_layer_edges, edges_feat, mask_valid, 
									  flat_indices=flat_indices, indices_reverse=indices_reverse)

		if channel_padding_mask is not None:
			nodes_out = nodes_out * channel_padding_mask
		edges_out = edges_out * mask_valid.unsqueeze(dim=-1)

		assert torch.isnan(nodes_out).sum() == 0 and torch.isnan(edges_out).sum() == 0, \
				"[!] ERROR: NaN output of EdgeGNN.\nNodes out: %s\nEdges out: %s" % (torch.isnan(nodes_out).sum().item(), torch.isnan(edges_out).sum().item())

		return nodes_out, edges_out



if __name__ == '__main__':
	torch.manual_seed(42)
	np.random.seed(42)

	def _generate_rand_input(batch_size, seq_len, c_in, num_edges=4, max_neighbours=4):
		z_in = torch.randn(batch_size, seq_len, c_in)
		length = torch.LongTensor(np.random.randint(2, seq_len+1, size=(batch_size,)))
		length[0] = seq_len
		adjacency = np.zeros((batch_size, seq_len, seq_len), dtype=np.int32)
		edge_probs = [1.8**(-e-1) for e in range(num_edges+1)]
		edge_probs = [e/sum(edge_probs) for e in edge_probs]
		for b in range(batch_size):
			for i in range(length[b]):
				for j in range(i+1,length[b]):
					if (adjacency[b,i,:] > 0).sum() >= max_neighbours or (adjacency[b,:,j] > 0).sum() >= max_neighbours:
						continue
					adjacency[b,i,j] = np.random.choice([int(e) for e in range(num_edges+1)], p=edge_probs)
					adjacency[b,j,i] = adjacency[b,i,j]
		adjacency = torch.LongTensor(adjacency)
		return z_in.to(get_device()), length.to(get_device()), adjacency.to(get_device())

	##################
	## Test RGCNNet ##
	##################
	def test_rgcnnet(debug=False):
		if not debug:
			c_in, hidden_size, c_out = 3, 512, 4 
			num_edges, max_neighbours = 3, 3
			batch_size, seq_len = 64, 40
		else:
			c_in, hidden_size, c_out = 3, 16, 4 
			num_edges, max_neighbours = 3, 3
			batch_size, seq_len = 2, 6
		module = RGCNNet(c_in=c_in, c_out=c_out, 
						 hidden_size=hidden_size, 
						 num_edges=num_edges, 
						 max_neighbours=max_neighbours, 
						 num_layers=1,
						 rgc_layer_fun=RelationGraphAttention
						 ).to(get_device())
		z_in, length, adjacency = _generate_rand_input(batch_size, seq_len, c_in, num_edges, max_neighbours)
		z_out = module(x=z_in, adjacency=adjacency)
		z_in[:,1,:] = -1
		z_out2 = module(x=z_in, adjacency=adjacency)

		if debug:
			print("adjacency\n", adjacency)
			print("Length", length)
			print("Z_out", z_out[:,:,0])
			print("Difference\n", (z_out-z_out2).abs())
		else:
			num_trials = 100
			start_time = time.time()
			for _ in range(num_trials):
				z_out = module(x=z_in, adjacency=adjacency)
			end_time = time.time()
			print("Average execution time: %5.3fs" % ((end_time - start_time) / num_trials))

	# test_rgcnnet()

	##################
	## Test EdgeGNN ##
	##################
	def test_edge_gnn(debug=False):
		if not debug:
			c_in_nodes, hidden_size_nodes, c_out_nodes = 4, 512, 8
			c_in_edges, hidden_size_edges, c_out_edges = 2, 256, 4
			batch_size, seq_len = 64, 38
			num_layers = 4
		else:
			c_in_nodes, hidden_size_nodes, c_out_nodes = 3, 8, 6
			c_in_edges, hidden_size_edges, c_out_edges = 2, 4, 4
			batch_size, seq_len = 1, 5
			num_layers = 1
		z_nodes = torch.randn(batch_size, seq_len, c_in_nodes).to(get_device())
		x_indices1 = torch.LongTensor([i for i in range(seq_len) for j in range(i+1, seq_len)]).to(get_device())
		x_indices2 = torch.LongTensor([j for i in range(seq_len) for j in range(i+1, seq_len)]).to(get_device())
		z_edges = torch.randn(batch_size, x_indices1.size(0), c_in_edges).to(get_device())
		length = torch.randint(2, seq_len, size=(batch_size,)).to(get_device())
		length[-1] = seq_len
		channel_padding_mask = (torch.arange(seq_len, dtype=torch.float32, device=length.device)[None,:,None] < length[:,None,None]).float()
		mask_valid = ((x_indices1[None,:] < length[:,None]) * (x_indices2[None,:] < length[:,None])).float()
		print("Efficiency", mask_valid.mean().item())

		edge2node_layer_func = lambda : Edge2NodeQKVAttnLayer(hidden_size_nodes=hidden_size_nodes, 
														   hidden_size_edges=hidden_size_edges,
														   skip_config=2)
		node2edge_layer_func = lambda : Node2EdgePlainLayer(hidden_size_nodes=hidden_size_nodes, 
															hidden_size_edges=hidden_size_edges,
															skip_config=2)
		edge_gnn_layer_func = lambda : EdgeGNNLayer(edge2node_layer_func=edge2node_layer_func, 
													node2edge_layer_func=node2edge_layer_func)
		module = EdgeGNN(c_in_nodes=c_in_nodes,
						 c_in_edges=c_in_edges,
						 c_out_nodes=c_out_nodes,
						 c_out_edges=c_out_edges,
						 edge_gnn_layer_func=edge_gnn_layer_func,
						 num_layers=num_layers)
		num_trials = 100 if not debug else 1
		module = module.to(get_device())
		start_time = time.time()
		for _ in range(num_trials):
			z_nodes_out, z_edges_out = module(z_nodes=z_nodes, z_edges=z_edges, length=length, x_indices=(x_indices1, x_indices2), mask_valid=mask_valid, channel_padding_mask=channel_padding_mask)
		end_time = time.time()
		max_memory = torch.cuda.max_memory_allocated(device=get_device())
		print("Average execution time: %5.3fs" % ((end_time - start_time) / num_trials))
		print("Maximum memory allocated: %4.2fGB" % (max_memory*1.0/1e9))
		print("Z nodes out", z_nodes_out[0,:10,0])
		print("Z edges out", z_edges_out[0,:10,0])
		if debug:
			print("Length", length)

	# test_edge_gnn()

	##################
	## Test EdgeGNN ##
	##################
	def test_edge_gnn_efficient(debug=False):
		if not debug:
			c_in_nodes, hidden_size_nodes, c_out_nodes = 4, 512, 8
			c_in_edges, hidden_size_edges, c_out_edges = 2, 256, 4
			batch_size, seq_len = 64, 38
			num_layers = 4
		else:
			c_in_nodes, hidden_size_nodes, c_out_nodes = 3, 8, 6
			c_in_edges, hidden_size_edges, c_out_edges = 2, 4, 4
			batch_size, seq_len = 2, 5
			num_layers = 1
		z_nodes, length, binary_adjacency = _generate_rand_input(batch_size, seq_len, c_in_nodes, num_edges=1, max_neighbours=3)
		x_indices1 = torch.LongTensor([i for i in range(seq_len) for j in range(i+1, seq_len)]).to(get_device())
		x_indices2 = torch.LongTensor([j for i in range(seq_len) for j in range(i+1, seq_len)]).to(get_device())
		z_edges = torch.randn(batch_size, x_indices1.size(0), c_in_edges).to(get_device())
		channel_padding_mask = (torch.arange(seq_len, dtype=torch.float32, device=length.device)[None,:,None] < length[:,None,None]).float()
		edge_pairs = binary_adjacency.view(batch_size, seq_len*seq_len).index_select(index=x_indices1+x_indices2*seq_len, dim=1)
		mask_valid = ((x_indices1[None,:] < length[:,None]) * (x_indices2[None,:] < length[:,None]) * (edge_pairs > 0)).float()
		if debug:
			print("Adjacency matrix", binary_adjacency.squeeze())
			print("Edge pairs", edge_pairs.squeeze())
			print("Efficiency", mask_valid.mean().item())
		
		# Edge2NodeAttnLayer, Edge2NodeQKVAttnLayer
		edge2node_layer_func = lambda : Edge2NodeQKVAttnLayer(hidden_size_nodes=hidden_size_nodes, 
														   hidden_size_edges=hidden_size_edges,
														   skip_config=2, num_heads=1)
		node2edge_layer_func = lambda : Node2EdgePlainLayer(hidden_size_nodes=hidden_size_nodes, 
															hidden_size_edges=hidden_size_edges,
															skip_config=2)
		edge_gnn_layer_func = lambda : EdgeGNNLayer(edge2node_layer_func=edge2node_layer_func, 
													node2edge_layer_func=node2edge_layer_func)
		module = EdgeGNN(c_in_nodes=c_in_nodes,
						 c_in_edges=c_in_edges,
						 c_out_nodes=c_out_nodes,
						 c_out_edges=c_out_edges,
						 edge_gnn_layer_func=edge_gnn_layer_func,
						 num_layers=num_layers)
		num_trials = 100 if not debug else 1
		module.eval()
		module = module.to(get_device())
		if debug:
			z_nodes_out1, z_edges_out1 = module(z_nodes=z_nodes, z_edges=z_edges, length=length, 
											  x_indices=(x_indices1, x_indices2), 
											  mask_valid=mask_valid, 
											  channel_padding_mask=channel_padding_mask, 
											  binary_adjacency=binary_adjacency
											  )
			z_nodes_out2, z_edges_out2 = module(z_nodes=z_nodes, z_edges=z_edges, length=length, 
											  x_indices=(x_indices1, x_indices2), 
											  mask_valid=mask_valid, 
											  channel_padding_mask=channel_padding_mask
											  )
			print("Z nodes out 1", z_nodes_out1[0,:10,0])
			print("Z nodes out 2", z_nodes_out2[0,:10,0])
			if batch_size > 1:
				z_nodes_out3, z_edges_out3 = module(z_nodes=z_nodes[0:1], z_edges=z_edges[0:1], length=length[0:1], 
											  x_indices=(x_indices1, x_indices2), 
											  mask_valid=mask_valid[0:1], 
											  channel_padding_mask=channel_padding_mask[0:1], 
											  binary_adjacency=binary_adjacency[0:1]
											  )
				print("Z nodes out 3", z_nodes_out3[0,:10,0])
		else:
			start_time = time.time()
			for _ in range(num_trials):
				z_nodes_out, z_edges_out = module(z_nodes=z_nodes, z_edges=z_edges, length=length, 
												  x_indices=(x_indices1, x_indices2), 
												  mask_valid=mask_valid, 
												  channel_padding_mask=channel_padding_mask, 
												  binary_adjacency=binary_adjacency
												  )
			end_time = time.time()
			max_memory = torch.cuda.max_memory_allocated(device=get_device())
			print("Average execution time: %5.3fs" % ((end_time - start_time) / num_trials))
			print("Maximum memory allocated: %4.2fGB" % (max_memory*1.0/1e9))
			print("Z nodes out", z_nodes_out[0,:10,0])
			print("Z edges out", z_edges_out[0,:10,0])

	test_edge_gnn_efficient(debug=False)

