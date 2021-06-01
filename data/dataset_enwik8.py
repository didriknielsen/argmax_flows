import torch
import os
import json
import zipfile
import urllib.request
from torch.utils.data import Dataset
from survae.data import TrainValidTestLoader, DATA_PATH
from .vocab import Vocab


class EnWik8(TrainValidTestLoader):
    def __init__(self, root=DATA_PATH, seq_len=256, download=True):
        self.train = EnWik8Dataset(root, seq_len=seq_len, split='train', download=download)
        self.valid = EnWik8Dataset(root, seq_len=seq_len, split='valid')
        self.test = EnWik8Dataset(root, seq_len=seq_len, split='test')


class EnWik8Dataset(Dataset):

    def __init__(self, root=DATA_PATH, seq_len=256, split='train', download=False):
        assert split in {'train', 'valid', 'test'}
        self.root = os.path.join(root, 'enwik8')
        self.seq_len = seq_len
        self.split = split

        if not os.path.exists(self.raw_file):
            if download:
                self.download()
            else:
                raise RuntimeError('Dataset not found. You can use download=True to download it.')

        # Get vocabulary
        self.vocab = Vocab()
        vocab_file = os.path.join(self.root, 'vocab.json')
        stoi = self._create_stoi()
        self.vocab.fill(stoi)

        # Load data
        self.data = self._preprocess_data(split).unsqueeze(1)

    def __getitem__(self, index):
        return self.data[index], self.seq_len

    def __len__(self):
        return len(self.data)

    def _create_stoi(self):
        # Just a simple identity conversion for 8bit (byte)-valued chunks.
        stoi = {i: i for i in range(256)}
        return stoi

    def _preprocess_data(self, split):
        # Read raw data
        rawdata = zipfile.ZipFile(self.raw_file).read('enwik8')

        n_train = int(90e6)
        n_valid = int(5e6)
        n_test = int(5e6)

        # Extract subset
        if split == 'train':
            rawdata = rawdata[:n_train]
        elif split == 'valid':
            rawdata = rawdata[n_train:n_train+n_valid]
        elif split == 'test':
            rawdata = rawdata[n_train+n_valid:n_train+n_valid+n_test]

        # Encode characters
        data = torch.tensor([self.vocab.stoi[s] for s in rawdata])

        # Split into chunks
        data = data.reshape(-1, self.seq_len)

        return data

    @property
    def raw_file(self):
        return os.path.join(self.root, 'enwik8.zip')

    def download(self):
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        print('Downloading enwik8...')
        url = 'http://mattmahoney.net/dc/enwik8.zip'
        print('Downloading from {}...'.format(url))
        urllib.request.urlretrieve(url, self.raw_file)
        print('Saved to {}'.format(self.raw_file))
