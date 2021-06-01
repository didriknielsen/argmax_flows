import torch 
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import math
from random import shuffle, random
from statistics import mean, median
import matplotlib.pyplot as plt
import os
import sys
import time
from statistics import stdev
sys.path.append("../")

from general.mutils import debug_level, append_in_dict, get_device


class TaskTemplate:


	def __init__(self, model, model_params, name, load_data=True, debug=False, batch_size=64, drop_last=False, num_workers=None):
		# Saving parameters
		self.name = name 
		self.model = model
		self.model_params = model_params
		self.batch_size = batch_size
		self.train_batch_size = batch_size
		self.debug = debug

		# Initializing dataset parameters
		self.train_dataset = None 
		self.val_dataset = None 
		self.test_dataset = None
		self.train_epoch = 0
		
		# Load data if specified, and create data loaders
		if load_data:
			self._load_datasets()
			self._initialize_data_loaders(drop_last=drop_last, num_workers=num_workers)
		else:
			self.train_data_loader = None 
			self.train_data_loader_iter = None
			self.val_data_loader = None 
			self.test_data_loader = None

		# Create a dictionary to store summary metrics in
		self.summary_dict = {}

		# Placeholders for visualization
		self.gen_batch = None
		self.class_colors = None

		# Put model on correct device
		self.model.to(get_device())


	def _initialize_data_loaders(self, drop_last, num_workers):
		if num_workers is None:
			if isinstance(self.model, nn.DataParallel) and torch.cuda.device_count() > 1:
				num_workers = torch.cuda.device_count()
			else:
				num_workers = 1

		def _init_fn(worker_id):
			np.random.seed(42)
		# num_workers = 1
		# Initializes all data loaders with the loaded datasets
		if hasattr(self.train_dataset, "get_sampler"):
			self.train_data_loader = data.DataLoader(self.train_dataset, batch_sampler=self.train_dataset.get_sampler(self.train_batch_size, drop_last=drop_last), pin_memory=True, 
													 num_workers=num_workers, worker_init_fn=_init_fn)
			self.val_data_loader = data.DataLoader(self.val_dataset, batch_sampler=self.val_dataset.get_sampler(self.train_batch_size, drop_last=False), pin_memory=True, num_workers=1, worker_init_fn=_init_fn)
			self.test_data_loader = data.DataLoader(self.test_dataset, batch_sampler=self.test_dataset.get_sampler(self.train_batch_size, drop_last=False), pin_memory=True, num_workers=1, worker_init_fn=_init_fn)		
		else:
			self.train_data_loader = data.DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, pin_memory=True, drop_last=drop_last, num_workers=num_workers,
													 worker_init_fn=_init_fn)
			self.val_data_loader = data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=1, worker_init_fn=_init_fn)
			self.test_data_loader = data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=1, worker_init_fn=_init_fn)
		self.train_data_loader_iter = iter(self.train_data_loader) # Needed to retrieve batch by batch from dataset
		

	def train_step(self, iteration=0):
		# Check if training data was correctly loaded
		if self.train_data_loader_iter is None:
			print("[!] ERROR: Iterator of the training data loader was None. Additional parameters: " + \
				  "train_data_loader was %sloaded, " % ("not " if self.train_data_loader is None else "") + \
				  "train_dataset was %sloaded." % ("not " if self.train_dataset is None else ""))
		
		# Get batch and put it on correct device
		batch = self._get_next_batch()
		batch = TaskTemplate.batch_to_device(batch)

		# Perform task-specific training step
		return self._train_batch(batch, iteration=iteration)


	def eval(self, data_loader=None, **kwargs):
		# Default: if no dataset is specified, we use validation dataset
		if data_loader is None:
			assert self.val_data_loader is not None, "[!] ERROR: Validation dataset not loaded. Please load the dataset beforehand for evaluation."
			data_loader = self.val_data_loader
		is_test = (data_loader == self.test_data_loader)

		start_time = time.time()
		torch.cuda.empty_cache()
		self.model.eval()
		
		# Prepare metrics
		total_nll, embed_nll, nll_counter = 0, 0, 0

		# Evaluation loop
		with torch.no_grad():
			for batch_ind, batch in enumerate(data_loader):
				if debug_level() == 0:
					print("Evaluation process: %4.2f%%" % (100.0 * batch_ind / len(data_loader)), end="\r")
				# Put batch on correct device
				batch = TaskTemplate.batch_to_device(batch)
				# Evaluate single batch
				batch_size = batch[0].size(0) if isinstance(batch, tuple) else batch.size(0)
				batch_nll = self._eval_batch(batch, is_test=is_test)
				total_nll += batch_nll.item() * batch_size
				nll_counter += batch_size

				if self.debug and batch_ind > 10:
					break

		avg_nll = total_nll / max(1e-5, nll_counter)
		detailed_metrics = {
			"negative_log_likelihood": avg_nll,
			"bpd": self.loss_to_bpd(avg_nll), # Bits per dimension
		}

		with torch.no_grad():
			self._eval_finalize_metrics(detailed_metrics, is_test=is_test, **kwargs)

		self.model.train()
		eval_time = int(time.time() - start_time)
		print("Finished %s with bpd of %4.3f (%imin %is)" % ("testing" if data_loader == self.test_data_loader else "evaluation", detailed_metrics["bpd"], eval_time/60, eval_time%60))
		torch.cuda.empty_cache()

		if "loss_metric" in detailed_metrics:
			loss_metric = detailed_metrics["loss_metric"]
		else:
			loss_metric = avg_nll
		
		return loss_metric, detailed_metrics


	def loss_to_bpd(self, loss):
		return (np.log2(np.exp(1)) * loss)


	def test(self, **kwargs):
		return self.eval(data_loader=self.test_data_loader, **kwargs)


	def add_summary(self, writer, iteration, checkpoint_path=None):
		# Adding metrics collected during training to the tensorboard
		# Function can/should be extended if needed
		for key, val in self.summary_dict.items():
			summary_key = "train_%s/%s" % (self.name, key)
			if not isinstance(val, list): # If it is not a list, it is assumably a single scalar
				writer.add_scalar(summary_key, val, iteration)
				self.summary_dict[key] = 0.0
			elif len(val) == 0: # Skip an empty list
				continue
			elif not isinstance(val[0], list): # For a list of scalars, report the mean
				writer.add_scalar(summary_key, mean(val), iteration)
				self.summary_dict[key] = list()
			else: # List of lists indicates a histogram
				val = [v for sublist in val for v in sublist]
				writer.add_histogram(summary_key, np.array(val), iteration)
				self.summary_dict[key] = list()


	def _ldj_per_layer_to_summary(self, ldj_per_layer, pre_phrase="ldj_layer_"):
		for layer_index, layer_ldj in enumerate(ldj_per_layer):
			if isinstance(layer_ldj, tuple) or isinstance(layer_ldj, list):
				for i in range(len(layer_ldj)):
					append_in_dict(self.summary_dict, "ldj_layer_%i_%i" % (layer_index, i), layer_ldj[i].detach().mean().item())
			elif isinstance(layer_ldj, dict):
				for key, ldj_val in layer_ldj.items():
					append_in_dict(self.summary_dict, "ldj_layer_%i_%s" % (layer_index, key), ldj_val.detach().mean().item())
			else:
				append_in_dict(self.summary_dict, "ldj_layer_%i" % layer_index, layer_ldj.detach().mean().item())


	def _get_next_batch(self):
		# Try to get next batch. If one epoch is over, the iterator throws an error, and we start a new iterator
		try:
			batch = next(self.train_data_loader_iter)
		except StopIteration:
			self.train_data_loader_iter = iter(self.train_data_loader)
			batch = next(self.train_data_loader_iter)
			self.train_epoch += 1
		return batch


	#######################################################
	### Abstract method to be implemented by subclasses ###
	#######################################################


	def _load_datasets(self):
		# Function for initializing datasets. Should set the following class parameters:
		# -> self.train_dataset
		# -> self.val_dataset
		# -> self.test_dataset
		raise NotImplementedError	


	def _train_batch(self, batch, iteration=0):
		# Given a batch, return the loss to be trained on
		raise NotImplementedError


	def _eval_batch(self, batch, is_test=False, take_mean=True):
		# Given a batch, return the negative log likelihood for its elements
		# Input arguments:
		# -> "is_test": True during testing, False during validation
		# -> "take_mean": If true, the return should be the average log 
		# 				  likelihood of the batch. Otherwise, return per element.
		raise NotImplementedError


	def finalize_summary(self, writer, iteration, checkpoint_path):
		# This function is called after the finishing training and performing testing.
		# Can be used to add something to the summary before finishing.
		pass


	def export_best_results(self, checkpoint_path, iteration):
		# This function is called if the last evaluation has been the best so far.
		# Can be used to add some output to the checkpoint directory.
		pass


	def initialize(self):
		# This function is called before starting the training.
		pass


	def _eval_finalize_metrics(self, detailed_metrics, is_test=False, initial_eval=False):
		# This function is called after evaluation finished. 
		# Can be used to add final metrics to the dictionary, which is added to the tensorboard.
		pass


	####################
	## Static methods ##
	####################

	@staticmethod
	def batch_to_device(batch):
		if isinstance(batch, tuple) or isinstance(batch, list):
			batch = tuple([b.to(get_device()) for b in batch])
		else:
			batch = batch.to(get_device())
		return batch