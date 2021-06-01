import os
import time
import math
import torch
import pickle
import argparse
import numpy as np
import torchvision.utils as vutils

# Data
from data.data import get_data, get_data_id, add_data_args

# Model
from model.model import get_model, get_model_id, add_model_args
from survae.distributions import DataParallelDistribution

###########
## Setup ##
###########

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--double', type=eval, default=False)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
eval_args = parser.parse_args()

path_args = '{}/args.pickle'.format(eval_args.model)
path_check = '{}/check/checkpoint.pt'.format(eval_args.model)

torch.manual_seed(eval_args.seed)

###############
## Load args ##
###############

with open(path_args, 'rb') as f:
    args = pickle.load(f)

##################
## Specify data ##
##################

train_loader, eval_loader, data_shape, num_classes = get_data(args)

###################
## Specify model ##
###################

model = get_model(args, data_shape=data_shape, num_classes=num_classes)
if args.parallel == 'dp':
    model = DataParallelDistribution(model)
checkpoint = torch.load(path_check)
model.load_state_dict(checkpoint['model'])
print('Loaded weights for model at {}/{} epochs'.format(checkpoint['current_epoch'], args.epochs))

############
## Sample ##
############

times = []

for seed in range(eval_args.seed, eval_args.seed+11):
    torch.manual_seed(seed)

    path_samples = os.path.join(eval_args.model, 'samples/sample_ep{}_s{}.txt'.format(checkpoint['current_epoch'], seed))
    if not os.path.exists(os.path.dirname(path_samples)):
        os.mkdir(os.path.dirname(path_samples))

    model = model.to(eval_args.device)
    model = model.eval()
    if eval_args.double: model = model.double()

    torch.cuda.synchronize()
    t0 = time.time()
    samples = model.sample(1)
    torch.cuda.synchronize()
    t1 = time.time()
    print('Time: {:.3f}'.format(t1-t0))
    times.append(t1-t0)
    samples = samples.cpu()
    samples_text = train_loader.dataset.tensor2text(samples)
    with open(path_samples, 'w') as f:
        f.write('\n'.join(samples_text))

times = times[1:]

print('Times: {:.3f} \pm {:.3f}'.format(np.mean(times), np.std(times)))
path_timing = os.path.join(eval_args.model, 'samples/timing10.txt')
with open(path_timing, 'w') as f:
    f.write('Times: {:.3f} \pm {:.3f}'.format(np.mean(times), np.std(times)))
