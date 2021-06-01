import torch 
import torch.nn as nn
import argparse
import random
import numpy as np
import math
import datetime
import os, shutil
import sys
sys.path.append("../")
import json
import pickle
import time
import signal
from glob import glob

from tensorboardX import SummaryWriter

from general.mutils import *


class TrainTemplate:
	"""
	Template class to handle the training loop.
	Each experiment contains a experiment-specific training class inherting from this template class.
	"""


	def __init__(self, model_params, optimizer_params, batch_size, checkpoint_path, debug=False, name_prefix="", multi_gpu=False):
		self.batch_size = batch_size
		self.name_prefix = name_prefix.strip() # Remove possible spaces. Name is used for creating default checkpoint path
		self.model_params = model_params
		self.optimizer_params = optimizer_params
		## Load model
		self.model = self._create_model(model_params)
		if multi_gpu: # Testing for multi-gpu if selected
			num_gpus = torch.cuda.device_count()
			if num_gpus == 0:
				print("[#] WARNING: Multi-GPU training failed because no GPU was detected. Continuing with single GPU...")
			elif num_gpus == 1:
				print("[#] WARNING: Multi-GPU training failed because only a single GPU is available. Continuing with single GPU...")
			else:
				print("Preparing to use %i GPUs..." % (num_gpus))
				self.model = WrappedDataParallel(self.model)

		self.model = self.model.to(get_device())
		## Load task
		self.task = self._create_task(model_params, debug=debug)
		## Load optimizer and checkpoints
		self._create_optimizer(optimizer_params)
		self._prepare_checkpoint(checkpoint_path)


	def _create_model(self, model_params):
		# To be implemented by the inherting class
		raise NotImplementedError


	def _create_task(self, model_params, debug=False):
		# To be implemented by the inherting class
		raise NotImplementedError


	def _create_optimizer(self, optimizer_params):
		parameters_to_optimize = self.model.parameters()
		self.optimizer = create_optimizer_from_args(parameters_to_optimize, optimizer_params)
		self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, optimizer_params["lr_decay_step"], gamma=optimizer_params["lr_decay_factor"])
		self.lr_minimum = optimizer_params["lr_minimum"]


	def _prepare_checkpoint(self, checkpoint_path):
		if checkpoint_path is None:
			current_date = datetime.datetime.now()
			checkpoint_path = "checkpoints/%s%02d_%02d_%02d__%02d_%02d_%02d/" % ((self.name_prefix + "_") if len(self.name_prefix)>0 else "", current_date.day, current_date.month, current_date.year, current_date.hour, current_date.minute, current_date.second)
		if not os.path.exists(checkpoint_path):
			os.makedirs(checkpoint_path)
		self.checkpoint_path = checkpoint_path


	def train_model(self, max_iterations=1e6, loss_freq=50, eval_freq=2000, save_freq=1e5, max_gradient_norm=0.25, no_model_checkpoints=False):

		parameters_to_optimize = self.model.parameters()

		# Setup dictionary to save evaluation details in
		checkpoint_dict = self.load_recent_model()
		start_iter = get_param_val(checkpoint_dict, "iteration", 0, warning_if_default=False) # Iteration to start from
		evaluation_dict = get_param_val(checkpoint_dict, "evaluation_dict", dict(), warning_if_default=False) # Dictionary containing validation performances over time
		best_save_dict = get_param_val(checkpoint_dict, "best_save_dict", {"file": None, "metric": 1e6, "detailed_metrics": None, "test": None}, warning_if_default=False) # 
		best_save_iter = best_save_dict["file"]
		last_save = None if start_iter == 0 else self.get_checkpoint_filename(start_iter)
		if last_save is not None and not os.path.isfile(last_save):
			print("[!] WARNING: Could not find last checkpoint file specified as " + last_save)
			last_save = None
		test_NLL = None # Possible test performance determined in the end of the training

		# Initialize tensorboard writer
		writer = SummaryWriter(self.checkpoint_path)

		# Function for saving model. Add here in the dictionary necessary parameters that should be saved
		def save_train_model(iteration, only_weights=True):
			if no_model_checkpoints:
				return
			checkpoint_dict = {
				"iteration": iteration,
				"best_save_dict": best_save_dict,
				"evaluation_dict": evaluation_dict
			}
			self.save_model(iteration, checkpoint_dict, save_optimizer=not only_weights)

		# Function to export the current results to a txt file
		def export_result_txt():
			if best_save_iter is not None:
				with open(os.path.join(self.checkpoint_path, "results.txt"), "w") as f:
					f.write("Best validation performance: %s\n" % (str(best_save_dict["metric"])))
					f.write("Best iteration: %i\n" % int(str(best_save_iter).split("_")[-1].split(".")[0]))
					f.write("Best checkpoint: %s\n" % str(best_save_iter))
					f.write("Detailed metrics\n")
					for metric_name, metric_val in best_save_dict["detailed_metrics"].items():
						f.write("-> %s: %s\n" % (metric_name, str(metric_val)))
					if "test" in best_save_dict and best_save_dict["test"] is not None:
						f.write("Test - Detailed metrics\n")
						for metric_name, metric_val in best_save_dict["test"].items():
							f.write("[TEST] -> %s: %s\n" % (metric_name, str(metric_val)))
					f.write("\n")
				
		# "Trackers" are moving averages. We use them to log the loss and time needed per training iteration
		time_per_step = Tracker()
		train_losses = Tracker()

		# Try-catch if user terminates
		try:
			index_iter = -1
			self.model.eval()
			self.task.initialize()
			print("="*50 + "\nStarting training...\n"+"="*50)
			self.model.train()

			print("Performing initial evaluation...")
			self.model.eval()
			eval_NLL, detailed_scores = self.task.eval(initial_eval=True)
			self.model.train()
			write_dict_to_tensorboard(writer, detailed_scores, base_name="eval", iteration=start_iter)		
			
			for index_iter in range(start_iter, int(max_iterations)):
				
				# Training step
				start_time = time.time()
				loss = self.task.train_step(iteration=index_iter)
				self.optimizer.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(parameters_to_optimize, max_gradient_norm)
				self.optimizer.step()
				if self.optimizer.param_groups[0]['lr'] > self.lr_minimum:
					self.lr_scheduler.step()
				end_time = time.time()

				time_per_step.add(end_time - start_time)
				train_losses.add(loss.item())

				# Statement for detecting NaN values 
				if torch.isnan(loss).item():
					print("[!] ERROR: Loss is NaN!" + str(loss.item()))
				for name, param in self.model.named_parameters():
					if param.requires_grad:
						if torch.isnan(param).sum() > 0:
							print("[!] ERROR: Parameter %s has %s NaN values!\n" % (name, str(torch.isnan(param).sum())) + \
								  "Grad values NaN: %s.\n" % (str(torch.isnan(param.grad).sum()) if param.grad is not None else "no gradients") + \
								  "Grad values avg: %s.\n" % (str(param.grad.abs().mean()) if param.grad is not None else "no gradients") + \
								  "Last loss: %s" % (str(loss)))

				# Printing current loss etc. for debugging
				if (index_iter + 1) % loss_freq == 0:
					loss_avg = train_losses.get_mean(reset=True)
					bpd_avg = self.task.loss_to_bpd(loss_avg)
					train_time_avg = time_per_step.get_mean(reset=True)
					max_memory = torch.cuda.max_memory_allocated(device=get_device())/1.0e9 if torch.cuda.is_available() else -1
					print("Training iteration %i|%i (%4.2fs). Loss: %6.5f, Bpd: %6.4f [Mem: %4.2fGB]" % (index_iter+1, max_iterations, train_time_avg, loss_avg, bpd_avg, max_memory))
					writer.add_scalar("train/loss", loss_avg, index_iter + 1)
					writer.add_scalar("train/bpd", bpd_avg, index_iter + 1)
					writer.add_scalar("train/learning_rate", self.optimizer.param_groups[0]['lr'], index_iter+1)
					writer.add_scalar("train/training_time", train_time_avg, index_iter+1)

					self.task.add_summary(writer, index_iter + 1, checkpoint_path=self.checkpoint_path)
	

				# Performing evaluation every "eval_freq" steps
				if (index_iter + 1) % eval_freq == 0:
					self.model.eval()
					eval_NLL, detailed_scores = self.task.eval()
					self.model.train()

					write_dict_to_tensorboard(writer, detailed_scores, base_name="eval", iteration=index_iter+1)

					# If model performed better on validation than any other iteration so far => save it and eventually replace old model
					if eval_NLL < best_save_dict["metric"]:
						best_save_iter = self.get_checkpoint_filename(index_iter+1)
						best_save_dict["metric"] = eval_NLL
						best_save_dict["detailed_metrics"] = detailed_scores
						if not os.path.isfile(best_save_iter):
							print("Saving model at iteration " + str(index_iter+1))
							if best_save_dict["file"] is not None and os.path.isfile(best_save_dict["file"]):
								print("Removing checkpoint %s..." % best_save_dict["file"])
								os.remove(best_save_dict["file"])
							if last_save is not None and os.path.isfile(last_save):
								print("Removing checkpoint %s..." % last_save)
								os.remove(last_save)
							best_save_dict["file"] = best_save_iter
							last_save = best_save_iter
							save_train_model(index_iter+1)
						self.task.export_best_results(self.checkpoint_path, index_iter + 1)
						export_result_txt()
					evaluation_dict[index_iter + 1] = best_save_dict["metric"]

				# Independent of evaluation, the model is saved every "save_freq" steps. This prevents loss of information if model does not improve for a while
				if (index_iter + 1) % save_freq == 0 and not os.path.isfile(self.get_checkpoint_filename(index_iter+1)):
					save_train_model(index_iter + 1)
					if last_save is not None and os.path.isfile(last_save) and last_save != best_save_iter:
						print("Removing checkpoint %s..." % last_save)
						os.remove(last_save)
					last_save = self.get_checkpoint_filename(index_iter+1)
			## End training loop
			
			# Before testing, load best model and check whether its validation performance is in the right range (to prevent major loading issues)
			if not no_model_checkpoints and best_save_iter is not None:
				load_model(best_save_iter, model=self.model, optimizer=self.optimizer, lr_scheduler=self.lr_scheduler)
				eval_NLL, detailed_scores = self.task.eval()
				if eval_NLL != best_save_dict["metric"]:
					if abs(eval_NLL - best_save_dict["metric"]) > 1e-1:
						print("[!] WARNING: new evaluation significantly differs from saved one (%s vs %s)! Probably a mistake in the saving/loading part..." % (str(eval_NLL), str(best_save_dict["metric"])))
					else:
						print("[!] WARNING: new evaluation sligthly differs from saved one (%s vs %s)." % (str(eval_NLL), str(best_save_dict["metric"])))
			else:
				print("Using last model as no models were saved...")
			
			# Testing the trained model
			test_NLL, detailed_scores = self.task.ground_truth()
			print("="*50+"\nTest performance: %lf" % (test_NLL))
			detailed_scores["original_NLL"] = test_NLL
			best_save_dict["test"] = detailed_scores
			self.task.finalize_summary(writer, max_iterations, self.checkpoint_path)

		# If user terminates training early, replace last model saved per "save_freq" steps by current one
		except KeyboardInterrupt:
			if index_iter > 0:
				print("User keyboard interrupt detected. Saving model at step %i..." % (index_iter))
				save_train_model(index_iter + 1)
			else:
				print("User keyboard interrupt detected before starting to train.")
			if last_save is not None and os.path.isfile(last_save) and not any([val == last_save for _, val in best_save_dict.items()]):
				os.remove(last_save)

		export_result_txt()

		writer.close()


	def get_checkpoint_filename(self, iteration):
		checkpoint_file = os.path.join(self.checkpoint_path, 'checkpoint_' + str(iteration).zfill(7) + ".tar")
		return checkpoint_file


	def save_model(self, iteration, add_param_dict, save_embeddings=False, save_optimizer=True):
		checkpoint_file = self.get_checkpoint_filename(iteration)
		if isinstance(self.model, nn.DataParallel):
			model_dict = self.model.module.state_dict()
		else:
			model_dict = self.model.state_dict()
		
		checkpoint_dict = {
			'model_state_dict': model_dict
		}
		if save_optimizer:
			checkpoint_dict['optimizer_state_dict'] = self.optimizer.state_dict()
			checkpoint_dict['scheduler_state_dict'] = self.lr_scheduler.state_dict()
		checkpoint_dict.update(add_param_dict)
		torch.save(checkpoint_dict, checkpoint_file)


	def load_recent_model(self):
		checkpoint_dict = load_model(self.checkpoint_path, model=self.model, optimizer=self.optimizer, lr_scheduler=self.lr_scheduler)
		return checkpoint_dict


	def evaluate_model(self, checkpoint_model=None):
		## Function for evaluation/testing of a model

		# Load the "best" model by first loading the most recent one and determining the "best" model
		checkpoint_dict = self.load_recent_model()
		best_save_dict = get_param_val(checkpoint_dict, "best_save_dict", {"file": None, "metric": -1, "detailed_metrics": dict()}, warning_if_default=True) # 
		best_save_iter = best_save_dict["file"]
		if not os.path.isfile(best_save_iter):
			splits = best_save_iter.split("/")
			checkpoint_index = splits.index("checkpoints")
			best_save_iter = "/".join(splits[checkpoint_index:])
		if not os.path.isfile(best_save_iter):
			print("[!] WARNING: Tried to load best model \"%s\", but file does not exist" % (best_save_iter))
		else:
			load_model(best_save_iter, model=self.model)

		# Print saved information of performance on validation set
		print("\n" + "-"*100 + "\n")
		print("Best evaluation iteration", best_save_iter)
		print("Best evaluation metric", best_save_dict["metric"])
		print("Detailed metrics")
		for metric_name, metric_val in best_save_dict["detailed_metrics"].items():
			print("-> %s: %s" % (metric_name, str(metric_val)))
		print("\n" + "-"*100 + "\n")

		# Test model
		self.task.checkpoint_path = self.checkpoint_path
		eval_metric, detailed_metrics = self.task.ground_truth()

		# Print test results
		out_dict = {}
		print("Evaluation metric", eval_metric)
		print("Detailed metrics")
		for metric_name, metric_val in detailed_metrics.items():
			print("-> %s: %s" % (metric_name, str(metric_val)))
			out_dict[metric_name] = str(metric_val) if isinstance(metric_val, torch.Tensor) else metric_val
		print("\n" + "-"*100 + "\n")

		# Save test results externally
		with open(os.path.join(self.checkpoint_path, "eval_metrics.json"), "w") as f: 
			json.dump(out_dict, f, indent=4)



def get_default_train_arguments():
	parser = argparse.ArgumentParser()
	# Training parameters
	parser.add_argument("--seed", help="Seed to make experiments reproducable", type=int, default=42)
	parser.add_argument("--max_iterations", help="Maximum number of epochs to train.", type=int, default=1e6)
	parser.add_argument("--batch_size", help="Batch size used during training", type=int, default=64)
	parser.add_argument("--print_freq", help="In which frequency loss information should be printed. Default: 250 if args.cluster, else 2", type=int, default=-1)
	parser.add_argument("--eval_freq", help="In which frequency the model should be evaluated (in number of iterations). Default: 2000", type=int, default=2000)
	parser.add_argument("--save_freq", help="In which frequency the model should be saved (in number of iterations). Default: 10,000", type=int, default=1e4)
	parser.add_argument("--use_multi_gpu", help="Whether to use all GPUs available or only one.", action="store_true")
	# Arguments for loading and saving models.
	parser.add_argument("--restart", help="Does not load old checkpoints, and deletes those if checkpoint path is specified (including tensorboard file etc.)", action="store_true")
	parser.add_argument("--checkpoint_path", help="Folder(name) where checkpoints should be saved", type=str, default=None)
	parser.add_argument("--load_config", help="Tries to find parameter file in checkpoint path, and loads all given parameters from there", action="store_true")
	parser.add_argument("--no_model_checkpoints", help="If selected, no model checkpoints will be saved", action="store_true")
	parser.add_argument("--only_eval", help="If selected, no training is performed but only an evaluation will be executed.", action="store_true")
	# Controlling the output
	parser.add_argument("--cluster", help="Enable option if code is executed on cluster. Reduces output size", action="store_true")
	parser.add_argument("-d", "--debug", help="Whether debug output should be activated or not", action="store_true")
	parser.add_argument("--clean_up", help="Whether to remove all files after finishing or not", action="store_true")
	# Optimizer parameters
	parser.add_argument("--learning_rate", help="Learning rate of the optimizer", type=float, default=7.5e-4)
	parser.add_argument("--lr_decay_factor", help="Decay of learning rate of the optimizer, applied after \"lr_decay_step\" training iterations.", type=float, default=0.999975)
	parser.add_argument("--lr_decay_step", help="Number of steps after which learning rate should be decreased", type=float, default=1)
	parser.add_argument("--lr_minimum", help="Minimum learning rate that should be scheduled. Default: no limit.", type=float, default=0.0)
	parser.add_argument("--weight_decay", help="Weight decay of the optimizer", type=float, default=0.0)
	parser.add_argument("--optimizer", help="Which optimizer to use. 0: SGD, 1: Adam, 2: Adamax, 3: RMSProp, 4: RAdam, 5: Adam Warmup", type=int, default=4)
	parser.add_argument("--momentum", help="Apply momentum to SGD optimizer", type=float, default=0.0)
	parser.add_argument("--beta1", help="Value for beta 1 parameter in Adam-like optimizers", type=float, default=0.9)
	parser.add_argument("--beta2", help="Value for beta 2 parameter in Adam-like optimizers", type=float, default=0.999)
	parser.add_argument("--warmup", help="If Adam with Warmup is selected, this value determines the number of warmup iterations to use.", type=int, default=2000)

	return parser


def start_training(args, parse_args_to_params_fun, TrainClass):
	"""
	Function to start a training loop. 
	Parameters:
		args - Argument namespace object produced by parser.parse_args()
		parse_args_to_params_fun - Function that takes the "args" object as input, and returns
								   the parameters for the model and optimizer as two dictionaries
		TrainClass - Class to start the training with. Should inherit TrainTemplate
	"""
	if args.cluster:
		set_debug_level(2)
		loss_freq = 250
	else:
		set_debug_level(0)
		loss_freq = 2
		if args.debug:
			# To find possible errors easier, activate anomaly detection. Note that this slows down training
			torch.autograd.set_detect_anomaly(True) 

	if args.print_freq > 0:
		loss_freq = args.print_freq

	only_eval = args.only_eval

	if args.load_config:
		if args.checkpoint_path is None:
			print("[!] ERROR: Please specify the checkpoint path to load the config from.")
			sys.exit(1)
		debug = args.debug
		checkpoint_path = args.checkpoint_path
		args = load_args(args.checkpoint_path)
		args.clean_up = False
		args.checkpoint_path = checkpoint_path
		if only_eval:
			args.use_multi_gpu = False
			args.debug = debug

	# Setup training
	model_params, optimizer_params = parse_args_to_params_fun(args)
	trainModule = TrainClass(model_params=model_params,
							 optimizer_params=optimizer_params, 
							 batch_size=args.batch_size,
							 checkpoint_path=args.checkpoint_path, 
							 debug=args.debug,
							 multi_gpu=args.use_multi_gpu
							 )

	# Function for cleaning up the checkpoint directory
	def clean_up_dir():
		assert str(trainModule.checkpoint_path) not in ["/", "/home/", "/lhome/"], \
			   "[!] ERROR: Checkpoint path is \"%s\" and is selected to be cleaned. This is probably not wanted..." % str(trainModule.checkpoint_path)
		print("Cleaning up directory " + str(trainModule.checkpoint_path) + "...")
		for file_in_dir in sorted(glob(os.path.join(trainModule.checkpoint_path, "*"))):
			print("Removing file " + file_in_dir)
			try:
				if os.path.isfile(file_in_dir):
					os.remove(file_in_dir)
				elif os.path.isdir(file_in_dir): 
					shutil.rmtree(file_in_dir)
			except Exception as e:
				print(e)

	if args.restart and args.checkpoint_path is not None and os.path.isdir(args.checkpoint_path) and not only_eval:
		clean_up_dir()

	if not only_eval:
		# Save argument namespace object for loading/evaluation
		args_filename = os.path.join(trainModule.checkpoint_path, PARAM_CONFIG_FILE)
		with open(args_filename, "wb") as f:
			pickle.dump(args, f)

		# Start training
		trainModule.train_model(args.max_iterations, loss_freq=loss_freq, eval_freq=args.eval_freq, save_freq=args.save_freq, no_model_checkpoints=args.no_model_checkpoints)

		# Cleaning up the checkpoint directory afterwards if selected
		if args.clean_up:
			clean_up_dir()
			os.rmdir(trainModule.checkpoint_path)
	else:
		# Only evaluating the model. Should be combined with loading a model.
		# However, the recommended way of evaluating a model is by the "eval.py" file in the experiment folder(s).
		trainModule.evaluate_model()

