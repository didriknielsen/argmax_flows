import torch
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt
import sys
sys.path.append("../")

from general.mutils import get_param_val


class ParameterScheduler:

	def __init__(self, param_name=None):
		self.param_name = param_name

	def get(self, iteration):
		raise NotImplementedError

	def info(self):
		return self._scheduler_description() + \
			   " for parameter %s" % str(self.param_name) if self.param_name is not None else ""

	def _scheduler_description(self):
		raise NotImplementedError


class ConstantScheduler(ParameterScheduler):

	def __init__(self, const_val, param_name=None):
		super().__init__(param_name=param_name)
		self.const_val = const_val


	def get(self, iteration):
		return self.const_val


	def _scheduler_description(self):
		return "Constant Scheduler on value %s" % str(self.const_val)


class SlopeScheduler(ParameterScheduler):

	def __init__(self, start_val, end_val, stepsize, logit_factor=0, delay=0, param_name=None):
		super().__init__(param_name=param_name)
		self.start_val = start_val
		self.end_val = end_val
		self.logit_factor = logit_factor
		self.stepsize = stepsize
		self.delay = delay
		assert self.stepsize > 0


	def get(self, iteration):
		if iteration < self.delay:
			return self.start_val
		else:
			iteration = iteration - self.delay
			return self.get_val(iteration)


	def get_val(self, iteration):
		raise NotImplementedError


class SigmoidScheduler(SlopeScheduler):

	def __init__(self, start_val, end_val, logit_factor, stepsize, delay=0, param_name=None):
		super().__init__(start_val=start_val, 
						 end_val=end_val, 
						 logit_factor=logit_factor, 
						 stepsize=stepsize, 
						 delay=delay, 
						 param_name=param_name)


	def get_val(self, iteration):
		return self.start_val + (self.end_val - self.start_val) / (1.0 + np.exp(-self.logit_factor * (iteration-self.stepsize)))


	def _scheduler_description(self):
		return "Sigmoid Scheduler from %s to %s with logit factor %s and stepsize %s" % \
				(str(self.start_val), str(self.end_val), str(self.logit_factor), str(self.stepsize))


class LinearScheduler(SlopeScheduler):

	def __init__(self, start_val, end_val, stepsize, delay=0, param_name=None):
		super().__init__(start_val=start_val, 
						 end_val=end_val, 
						 logit_factor=0, 
						 stepsize=stepsize, 
						 delay=delay, 
						 param_name=param_name)


	def get_val(self, iteration):
		if iteration >= self.stepsize:
			return self.end_val
		else:
			return self.start_val + (self.end_val - self.start_val) * (iteration * 1.0 / self.stepsize)


	def _scheduler_description(self):
		return "Linear Scheduler from %s to %s in %s steps" % \
				(str(self.start_val), str(self.end_val), str(self.stepsize))


class ExponentialScheduler(SlopeScheduler):

	def __init__(self, start_val, end_val, logit_factor, stepsize, delay=0, param_name=None):
		super().__init__(start_val=start_val, 
						 end_val=end_val, 
						 logit_factor=logit_factor, 
						 stepsize=stepsize, 
						 delay=delay, 
						 param_name=param_name)


	def get_val(self, iteration):
		return self.start_val + (self.end_val - self.start_val) * (1 - self.logit_factor ** (-iteration*1.0/self.stepsize))


	def _scheduler_description(self):
		return "Exponential Scheduler from %s to %s with logit %s and stepsize %s" % \
				(str(self.start_val), str(self.end_val), str(self.logit_factor), str(self.stepsize))



def add_scheduler_parameters(parser, param_names, default_vals=None):
	for name in param_names:
		default_type, default_end_val, default_start_val, default_step_size, default_logit, default_delay = "constant", 0.0, 0.0, 100, 2.0, 0
		if default_vals is not None and name in default_vals:
			default_type, default_end_val, default_start_val, default_step_size, default_logit, default_delay = default_vals[name]
		parser.add_argument("--%s_scheduler_type" % name, help="Which type of scheduler to use for the parameter %s. Options: constant, sigmoid, exponential" % name, type=str, default=default_type)
		parser.add_argument("--%s_scheduler_end_val" % name, help="Value of the parameter %s which should be reached for t->infinity." % name, type=float, default=default_end_val)
		parser.add_argument("--%s_scheduler_start_val" % name, help="Value of the parameter %s at t=0" % name, type=float, default=default_start_val)
		parser.add_argument("--%s_scheduler_step_size" % name, help="Step size which should be used in the scheduler for %s." % name, type=int, default=default_step_size)
		parser.add_argument("--%s_scheduler_logit" % name, help="Logit which should be used in the scheduler for %s." % name, type=float, default=default_logit)
		parser.add_argument("--%s_scheduler_delay" % name, help="Delay which should be used in the scheduler for %s." % name, type=int, default=default_delay)
	return parser


def scheduler_args_to_params(args, param_names):
	params = {}
	for name in param_names:
		params[name] = {
			p: getattr(args, "%s_%s" % (name, p)) for p in \
				["scheduler_type", "scheduler_end_val", "scheduler_start_val", "scheduler_step_size", "scheduler_logit", "scheduler_delay"]
		}
	return params


def create_scheduler(scheduler_params, param_name=None):
	sched_type = get_param_val(scheduler_params, "scheduler_type", allow_default=False)
	end_val = get_param_val(scheduler_params, "scheduler_end_val", allow_default=False)
	start_val = get_param_val(scheduler_params, "scheduler_start_val", allow_default=False)
	stepsize = get_param_val(scheduler_params, "scheduler_step_size", allow_default=False)
	logit = get_param_val(scheduler_params, "scheduler_logit", allow_default=False)
	delay = get_param_val(scheduler_params, "scheduler_delay", allow_default=False)

	if sched_type == "constant":
		return ConstantScheduler(const_val=end_val, param_name=param_name)
	elif sched_type == "linear":
		return LinearScheduler(start_val=start_val, end_val=end_val, stepsize=stepsize, delay=delay, param_name=param_name)
	elif sched_type == "sigmoid":
		return SigmoidScheduler(start_val=start_val, end_val=end_val, logit_factor=logit, stepsize=stepsize, delay=delay, param_name=param_name)
	elif sched_type == "exponential":
		return ExponentialScheduler(start_val=start_val, end_val=end_val, logit_factor=logit, stepsize=stepsize, delay=delay, param_name=param_name)
	else:
		print("[!] ERROR: Unknown scheduler type \"%s\"" % str(sched_type))
		sys.exit(1)


def visualize_scheduler(scheduler, max_iterations=10000):
	iterations = range(max_iterations)
	vals = [scheduler.get(i) for i in iterations]
	plt.plot(iterations, vals)
	plt.title(scheduler.info())
	plt.xlabel("Iterations")
	plt.ylabel("Value")
	plt.grid()
	plt.show()


if __name__ == '__main__':
	scheduler = SigmoidScheduler(start_val=0.1, end_val=1.0, logit_factor=0.0003, stepsize=15000, delay=0, param_name="Gamma")
	visualize_scheduler(scheduler, max_iterations=30000)