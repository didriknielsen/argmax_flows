import torch
import torch.utils.data as data
import os
import urllib.request
import zipfile
import json
from survae.data import TrainValidTestLoader, DATA_PATH


class Text8(TrainValidTestLoader):
    def __init__(self, root=DATA_PATH, seq_len=256, download=True):
        self.train = Text8Dataset(root, seq_len=seq_len, split='train', download=download)
        self.valid = Text8Dataset(root, seq_len=seq_len, split='valid')
        self.test = Text8Dataset(root, seq_len=seq_len, split='test')


class Text8Dataset(data.Dataset):
    """
    The text8 dataset consisting of 100M characters (with vocab size 27).
    We here split the dataset into (90M, 5M, 5M) characters for
    (train, val, test) as in [1,2,3].
    The sets are then split into chunks of equal length as specified by `seq_len`.
    The default is 256, corresponding to what was used in [1]. Other choices
    include 180, as [2] reports using.
    [1] Discrete Flows: Invertible Generative Models of Discrete Data
        Tran et al., 2019, https://arxiv.org/abs/1905.10347
    [2] Architectural Complexity Measures of Recurrent Neural Networks
        Zhang et al., 2016, https://arxiv.org/abs/1602.08210
    [3] Subword Language Modeling with Neural Networks
        Mikolov et al., 2013, http://www.fit.vutbr.cz/~imikolov/rnnlm/char.pdf
    """

    def __init__(self, root=DATA_PATH, seq_len=256, split='train', download=False):
        assert split in {'train', 'valid', 'test'}
        self.root = os.path.join(os.path.expanduser(root), 'text8')
        self.seq_len = seq_len
        self.split = split

        if not self._check_raw_exists():
            if download:
                self.download()
            else:
                raise RuntimeError('Dataset not found. You can use download=True to download it.')

        if not self._check_processed_exists(split):
            self._preprocess_data(split)

        # Load data
        self.data = torch.load(self.processed_filename(split))

        # Load lookup tables
        char2idx_file = os.path.join(self.root, 'char2idx.json')
        idx2char_file = os.path.join(self.root, 'idx2char.json')
        with open(char2idx_file) as f:
            self.char2idx = json.load(f)
        with open(idx2char_file) as f:
            self.idx2char = json.load(f)

    def __getitem__(self, index):
        return self.data[index], self.seq_len

    def __len__(self):
        return len(self.data)

    def s2t(self, s):
        assert len(s) == self.seq_len, 'String not of length {}'.format(self.seq_len)
        return torch.tensor([self.char2idx[char] for char in s])

    def t2s(self, t):
        return ''.join([self.idx2char[t[i]] if t[i] < len(self.idx2char) else ' ' for i in range(self.seq_len)])

    def text2tensor(self, text):
        if isinstance(text, str):
            tensor = self.s2t(text).unsqueeze(0)
        else:
            tensor = torch.stack([self.s2t(s) for s in text], dim=0)
        return tensor.unsqueeze(1) # (B, 1, L)

    def tensor2text(self, tensor):
        assert tensor.dim() == 3, 'Tensor should have shape (batch_size, 1, {})'.format(self.seq_len)
        assert tensor.shape[1] == 1, 'Tensor should have shape (batch_size, 1, {})'.format(self.seq_len)
        assert tensor.shape[2] == self.seq_len, 'Tensor should have shape (batch_size, 1, {})'.format(self.seq_len)
        bsize = tensor.shape[0]
        text = [self.t2s(tensor[b].squeeze(0)) for b in range(bsize)]
        return text

    def _preprocess_data(self, split):
        # Read raw data
        rawdata = zipfile.ZipFile(self.local_filename).read('text8').decode('utf-8')

        # Extract vocab
        vocab = sorted(list(set(rawdata)))
        char2idx, idx2char = {}, []
        for i, char in enumerate(vocab):
            char2idx[char] = i
            idx2char.append(char)

        # Extract subset
        if split == 'train':
            rawdata = rawdata[:90000000]
        elif split == 'valid':
            rawdata = rawdata[90000000:95000000]
        elif split == 'test':
            rawdata = rawdata[95000000:]

        # Encode characters
        data = torch.tensor([char2idx[char] for char in rawdata])

        # Split into chunks
        data = data[:self.seq_len*(len(data)//self.seq_len)]
        data = data.reshape(-1, 1, self.seq_len)

        # Save processed data
        torch.save(data, self.processed_filename(split))

        # Save lookup tables
        char2idx_file = os.path.join(self.root, 'char2idx.json')
        idx2char_file = os.path.join(self.root, 'idx2char.json')
        with open(char2idx_file, 'w') as f:
            json.dump(char2idx, f)
        with open(idx2char_file, 'w') as f:
            json.dump(idx2char, f)

    @property
    def local_filename(self):
        return os.path.join(self.root, 'text8.zip')

    def processed_filename(self, split):
        return os.path.join(self.root, '{}.pt'.format(split))

    def download(self):
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        print('Downloading text8...')

        url = 'http://mattmahoney.net/dc/text8.zip'
        print('Downloading from {}...'.format(url))
        urllib.request.urlretrieve(url, self.local_filename)
        print('Saved to {}'.format(self.local_filename))

    def _check_raw_exists(self):
        return os.path.exists(self.local_filename)

    def _check_processed_exists(self, split):
        return os.path.exists(self.processed_filename(split))
