from __future__ import print_function, division
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils import ConfigBase, Logger
import pdb
from kaldi.matrix import Vector, Matrix, DoubleVector, DoubleMatrix
from kaldi.util.table import VectorWriter, SequentialVectorReader
import kaldi_io
from random import choice, random
import numpy as np


class Config(ConfigBase):
    def __init__(self):
        super(Config, self).__init__()
        self.data_loader_config = "======Ivector Preprocess Configuration======="
        self.normalize_length = False
        self.ivector_dim = 600
        # self.spk2utt_scp = "ark:data/train/spk2utt"
        # self.ivector_scp = "scp:exp/ivectors_train/ivector.scp"
        self.neg_pos_ratio = 10


def scp2dict(ipath2scp):
    fd = kaldi_io.open_or_fd(ipath2scp)  # iapth2scp can be a pipeline
    id2path = {}
    for line in fd:
        items = line.decode("utf-8").rstrip().split(' ')
        id = items[0]
        id2path[id] = items[1:]
    return id2path


class PairwiseIvectorDataset(Dataset):
    """
        Generate pairs of ivector for training
    """
    def __init__(self, spk2utt_scp, ivector_scp, config):
        spk2utt = scp2dict(spk2utt_scp)
        self.normalize_length = config.normalize_length
        self.ivector_dim = config.ivector_dim
        self.neg_pos_ratio = config.neg_pos_ratio
        
        utt2idx, idx2utt, ivecs = self.format_ivectors(
                    ivector_scp, self.ivector_dim, self.normalize_length)
        pairs = self.generate_pairs(spk2utt, utt2idx, self.neg_pos_ratio)

        self.spk2utt = spk2utt  # this is just for debug
        self.pairs = pairs
        self.utt2idx = utt2idx
        self.idx2utt = idx2utt
        self.ivecs = ivecs

    def format_ivectors(self, ivector_scp, ivector_dim, normalize_length=False):
        utt2ivecpath = scp2dict(ivector_scp)
        num_utts = len(utt2ivecpath)
        utt2idx = dict()
        idx2utt = []
        ivecs = np.empty(shape=(num_utts, ivector_dim))
        ivector_cmd = ivector_scp

        if normalize_length:
            ivector_cmd = ("ark:ivector-normalize-length {ivector_scp} ark:- |").format(
                   ivector_scp=ivector_scp)
        with SequentialVectorReader(ivector_cmd) as reader:
            for i, (utt, ivector) in enumerate(reader):
                utt2idx[utt] = i
                idx2utt.append(utt)
                ivecs[i] = ivector.numpy()

        return utt2idx, idx2utt, ivecs

    def generate_pairs(self, spk2utt, utt2idx, neg_pos_ratio=10):
        pairs = []
        spks = list(spk2utt.keys())
        num_spks = len(spks)
        
        # genereate pairs that belong to the same speaker
        for spk in spks:
            utts = spk2utt[spk]
            num_utts = len(utts)
            if num_utts < 2:
                continue
            for i in range(num_utts-1):
                for j in range(i, num_utts):
                    pairs.append((utt2idx[utts[i]], utt2idx[utts[j]], 1))

        assert num_spks > 1, "Only {0} speaker in the dataset ?".format(num_spks)
        num_same_pairs = len(pairs)
        print("Created {0} same-pairs from {1} speakers".format(num_same_pairs, num_spks))
        # num is #{random selected utterances} from each pair of different speakers
        num = len(pairs) * neg_pos_ratio * 2 / (num_spks * (num_spks -1)) # is it int ?
        num = int(num+1)
        # genereate pairs that belong to different speakers
        for spk_i in range(num_spks-1):
            for spk_j in range(spk_i, num_spks):
                utts_i = spk2utt[spks[spk_i]]
                utts_j = spk2utt[spks[spk_j]]
                utts_i_num = len(utts_i) - 1
                utts_j_num = len(utts_j) - 1
                for i in range(num):
                    # utti = choice(utts_i)
                    # uttj = choice(utts_j)
                    utti = utts_i[int(random() * utts_i_num + 0.499)]
                    uttj = utts_j[int(random() * utts_j_num + 0.499)]
                    pairs.append((utt2idx[utti], utt2idx[uttj], -1))
        print("Created {0} diff-pairs with {1} pairs in total".format(
            len(pairs)-num_same_pairs, len(pairs)))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        return self.pairs[index]


if __name__ == '__main__':
    config = Config()
    config.parse_args() # revise configurations from cmd line
    config.print_args()
    dataset = PairwiseIvectorDataset(
        "ark:data/train/spk2utt",
        "scp:exp/ivectors_train/ivector.scp",
        config)
    dataset[0]

    # train_loader = DataLoader(dataset=dataset, batch_size=128, shuffle=True, num_workers=2, drop_last=True)
        # self.spk2utt_scp = "ark:data/train/spk2utt"
        # self.ivector_scp = "scp:exp/ivectors_train/ivector.scp"
    # sys.stdout = Logger('{0}/dataloader.log'.format(config.log_dir))

