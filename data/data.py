import torch
from torch.utils.data import DataLoader, ConcatDataset
from .dataset_text8 import Text8
from .dataset_enwik8 import EnWik8

dataset_choices = {'text8_f256', 'enwik8_f320'}

def add_data_args(parser):

    # Data params
    parser.add_argument('--dataset', type=str, default='text8_f256', choices=dataset_choices)
    parser.add_argument('--validation', type=eval, default=True)

    # Train params
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=eval, default=False)


def get_data_id(args):
    return args.dataset


def get_data(args):
    assert args.dataset in dataset_choices

    # Dataset
    if args.dataset == 'text8_f256':
        data = Text8(seq_len=256)
        data_shape = (1,256)
        num_classes = 27
    elif args.dataset == 'enwik8_f320':
        data = EnWik8(seq_len=320)
        data_shape = (1,320)
        num_classes = 256

    # Data Loader
    if args.validation:
        train_loader = DataLoader(data.train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
        eval_loader = DataLoader(data.valid, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    else:
        dataset_train = ConcatDataset([data.train, data.valid])
        train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
        eval_loader = DataLoader(data.test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    return train_loader, eval_loader, data_shape, num_classes
