# Code for Argmax Flows

![Banner](images/overview_argmax.png?raw=true)

## Overview

> Official code for [Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions](https://arxiv.org/abs/2102.05379)  
by Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forré, Max Welling.

Code for **Argmax Flows**: This repo.  
Code for  **Multinomal Diffusion**: See [here](https://github.com/ehoogeboom/multinomial_diffusion).

## Running Experiments

This code depends on the `survae` library, which is available at https://github.com/didriknielsen/survae_flows.

**To reproduce experiments run:**

Autoregressive `text8`:
```
python train.py --lstm_size 2048 --encoder_steps 0 --num_steps 1 --batch_size 64 --lstm_layers 2 --epochs 40 --eval_every 1 --check_every 10 --input_dp_rate 0.25 --model ar --dataset text8_f256
```

Autoregressive `enwik8`:  
```  
python train.py --lstm_size 2048 --encoder_steps 0 --num_steps 1 --batch_size 64 --lstm_layers 2 --epochs 40 --eval_every 1 --check_every 10 --input_dp_rate 0.25 --model ar --dataset enwik8_f320
```

Coupling `text8`:
```
python train.py --lstm_size 512 --encoder_steps 0 --num_steps 8 --batch_size 16 --lstm_layers 2 --epochs 40 --eval_every 1 --check_every 1 --input_dp_rate 0.05 --parallel dp --optimizer adamax --lr 1e-3 --warmup 1000 --num_mixtures 8 --model coupling --dataset text8_f256
```

Coupling `enwik8`:  
```
python train.py --lstm_size 768 --encoder_steps 0 --num_steps 8 --batch_size 32 --lstm_layers 2 --epochs 20 --eval_every 1 --check_every 1 --input_dp_rate 0.1 --parallel dp --optimizer adamax --lr 1e-3 --gamma 0.95 --warmup 1000 --num_mixtures 8 --model coupling --dataset enwik8_f320
```

## Acknowledgements
The Robert Bosch GmbH is acknowledged for financial support.
