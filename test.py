from __future__ import print_function, division
import pdb
import numpy as np
from torch.utils.data import Dataset
from dataloader import scp2dict
from utils import ConfigBase, compute_eer
from cmodel import PSVM
from kaldi.util.table import SequentialVectorReader
import kaldi_io
from random import random
import datetime
import time
import random
import os



class Config(ConfigBase):
    def __init__(self):
        super(Config, self).__init__()
        self.test_config = "====Test Pairwise SVM Configuration===="
        self.log_dir = "exp/psvm/log"
        self.ipath2model = "exp/psvm/psvm.mdl"
        # self.test_spk2utt = "ark:data/test/spk2utt"
        self.trials = "data/voxceleb1_trials_sv"
        self.test_ivector_scp = "scp:exp/ivectors_test/ivector.scp"
        self.opath2score = "exp/psvm/score/rpsvm.score"
        self.normalize_length = False
        self.ivector_dim = 600


class TestIvectorDataset(Dataset):
    """
        Generate pairs of ivector for evaluation
    """

    def __init__(self, config):
        self.trials = config.trials
        self.test_ivector_scp = config.test_ivector_scp
        self.normalize_length = config.normalize_length
        self.ivector_dim = config.ivector_dim

        utt2idx, idx2utt, ivecs = self.format_ivectors(
            self.test_ivector_scp, self.ivector_dim, self.normalize_length)
        pairs = self.generate_pairs(self.trials, utt2idx)

        self.utt2idx = utt2idx
        self.idx2utt = idx2utt
        self.ivecs = ivecs
        self.pairs = pairs

    def format_ivectors(self, ivector_scp, ivector_dim, normalize_length=False):
        num_utts = len(scp2dict(ivector_scp))
        utt2idx = dict()
        idx2utt = []
        ivecs = np.empty(shape=(num_utts, ivector_dim))
        ivector_cmd = ivector_scp

        if normalize_length:
            ivector_cmd = (
                "ark:ivector-normalize-length {ivector_scp} ark:- |").format(ivector_scp=ivector_scp)
        with SequentialVectorReader(ivector_cmd) as reader:
            for i, (utt, ivector) in enumerate(reader):
                utt2idx[utt] = i
                idx2utt.append(utt)
                ivecs[i] = ivector.numpy()

        return utt2idx, idx2utt, ivecs

    def generate_pairs(self, trials, utt2idx):
        pairs = []
        with open(trials, 'r') as fin:
            for line in fin:
                # pdb.set_trace()
                utt1, utt2 = line.rstrip('\n').split(' ')[:2]
                pairs.append((utt2idx[utt1], utt2idx[utt2]))
        print("Read in {0} pairs in total".format(len(pairs)))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        return self.pairs[index]


def main(config):
    testIvectorDataset = TestIvectorDataset(config)
    pSVM = PSVM(testIvectorDataset.ivecs)
    pSVM.load_model(config.ipath2model)
    pSVM.scoring(testIvectorDataset, testIvectorDataset.idx2utt,
                 config.opath2score)
    compute_eer(config.opath2score, config.trials)


if __name__ == '__main__':
    config = Config()
    config.parse_args() # revise configuration from cmd line
    config.print_args()
    if not os.path.exists(os.path.dirname(config.opath2score)):
        os.makedirs(os.path.dirname(config.opath2score))
    main(config)

