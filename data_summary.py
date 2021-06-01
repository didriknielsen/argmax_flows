import torch
import argparse

# Plot
import matplotlib.pyplot as plt

# Data
from data.data import get_data, add_data_args

###########
## Setup ##
###########

parser = argparse.ArgumentParser()
add_data_args(parser)
args = parser.parse_args()

torch.manual_seed(0)

##################
## Specify data ##
##################

train_loader, eval_loader, data_shape, num_classes = get_data(args)

##############
## Sampling ##
##############

print('Train Batches:', len(train_loader))
batch, lengths = next(iter(train_loader))
print(batch.shape, batch.min(), batch.max())
print(lengths)
print('Sample 0:')
print(batch[0,0])
if hasattr(train_loader.dataset, 'tensor2text'):
    print(train_loader.dataset.tensor2text(batch[0].unsqueeze(0)))
