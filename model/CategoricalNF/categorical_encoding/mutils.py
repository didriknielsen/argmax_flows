import torch
import torch.nn as nn 
import torch.nn.functional as F
import sys
sys.path.append("../")

from ..general.mutils import get_param_val
from .variational_dequantization import VariationalDequantization
from .linear_encoding import LinearCategoricalEncoding
from .variational_encoding import VariationalCategoricalEncoding



def add_encoding_parameters(parser, postfix=""):
	# General parameters
	parser.add_argument("--encoding_dim" + postfix, help="Dimensionality of the embeddings.", type=int, default=4)
	parser.add_argument("--encoding_dequantization" + postfix, help="If selected, variational dequantization is used for encoding categorical data.", action="store_true")
	parser.add_argument("--encoding_variational" + postfix, help="If selected, the encoder distribution is joint over categorical variables.", action="store_true")
	
	# Flow parameters
	parser.add_argument("--encoding_num_flows" + postfix, help="Number of flows used in the embedding layer.", type=int, default=0)
	parser.add_argument("--encoding_hidden_layers" + postfix, help="Number of hidden layers of flows used in the parallel embedding layer.", type=int, default=2)
	parser.add_argument("--encoding_hidden_size" + postfix, help="Hidden size of flows used in the parallel embedding layer.", type=int, default=128)
	parser.add_argument("--encoding_num_mixtures" + postfix, help="Number of mixtures used in the coupling layers (if applicable).", type=int, default=8)
	
	# Decoder parameters
	parser.add_argument("--encoding_use_decoder" + postfix, help="If selected, we use a decoder instead of calculating the likelihood by inverting all flows.", action="store_true")
	parser.add_argument("--encoding_dec_num_layers" + postfix, help="Number of hidden layers used in the decoder of the parallel embedding layer.", type=int, default=1)
	parser.add_argument("--encoding_dec_hidden_size" + postfix, help="Hidden size used in the decoder of the parallel embedding layer.", type=int, default=64)


def encoding_args_to_params(args, postfix=""):
	params = {
		"use_dequantization": getattr(args, "encoding_dequantization" + postfix),
		"use_variational": getattr(args, "encoding_variational" + postfix),
		"use_decoder": getattr(args, "encoding_use_decoder" + postfix),
		"num_dimensions": getattr(args, "encoding_dim" + postfix), 
		"flow_config": {
			"num_flows": getattr(args, "encoding_num_flows" + postfix),
			"hidden_layers": getattr(args, "encoding_hidden_layers" + postfix),
			"hidden_size": getattr(args, "encoding_hidden_size" + postfix)
		},
		"decoder_config": {
			"num_layers": getattr(args, "encoding_dec_num_layers" + postfix),
			"hidden_size": getattr(args, "encoding_dec_hidden_size" + postfix)
		}
	}
	return params


def create_encoding(encoding_params, dataset_class, vocab=None, vocab_size=-1, category_prior=None):
	assert not (vocab is None and vocab_size <= 0), "[!] ERROR: When creating the encoding, either a torchtext vocabulary or the vocabulary size needs to be passed."
	use_dequantization = encoding_params.pop("use_dequantization")
	use_variational = encoding_params.pop("use_variational")


	if use_dequantization and "model_func" not in encoding_params["flow_config"]:
		print("[#] WARNING: For using variational dequantization as encoding scheme, a model function needs to be specified" + \
			  " in the encoding parameters, key \"flow_config\" which was missing here. Will deactivate dequantization...")
		use_dequantization = False

	if use_dequantization:
		encoding_flow = VariationalDequantization
	elif use_variational:
		encoding_flow = VariationalCategoricalEncoding
	else:
		encoding_flow = LinearCategoricalEncoding

	print('HALLLO WAT GAAN WE GEBRUIKEN', encoding_flow)

	return encoding_flow(dataset_class=dataset_class,
						 vocab=vocab,
						 vocab_size=vocab_size,
						 category_prior=category_prior,
						 **encoding_params)