import pdb
import numpy as np
from scipy.sparse import csc_matrix
from dataloader import Config as DataLoaderConfig
import datetime
import time
import random
import os
from cython.parallel import prange
from libc.stdlib cimport abort, malloc, free
cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


class PSVM(object):
    def __init__(self, ivecs):
        super(PSVM, self).__init__()
        ivec_num, ivec_dim = ivecs.shape
        assert ivec_dim > 1, "Invalid arg. ivec_dim ({}) should > 1".format(
            ivec_dim)
        assert ivec_num > 1, "Invalid arg. ivec_num ({}) should > 1".format(
            ivec_num)
        self.ivec_num = ivec_num
        self.ivec_dim = ivec_dim
        self.ivecs = ivecs
        # model paras
        self.Lambda = np.random.rand(ivec_dim, ivec_dim)
        self.Gamma = np.random.rand(ivec_dim, ivec_dim)
        self.c = np.random.rand(ivec_dim)  # ivector
        self.k = np.random.rand()

        # intermediate produced by precompute
        self.Lambda_x = np.random.rand(ivec_num, ivec_dim)
        self.xt_Gamma_x = np.random.rand(ivec_num)
        self.xt_c = np.random.rand(ivec_num)

        # scores, sum_sv_y
        self.reset_accs()

    def precompute(self):
        x = self.ivecs
        assert x.shape == (self.ivec_num, self.ivec_dim), "dim of x: ({}, {}) doesn't match ({}, {})".format(
            x.shape[0], x.shape[1], self.ivec_num, self.ivec_dim)
        # pdb.set_trace()
        self.Lambda_x = np.dot(x, self.Lambda)
        self.xt_Gamma_x = np.dot(x, self.Gamma)
        self.xt_Gamma_x = np.einsum('ij,ij->i', self.xt_Gamma_x, x)
        self.xt_c = np.dot(x, self.c)

    def reset_accs(self):
        # self.G = csc_matrix((self.ivec_num, self.ivec_num))
        self.stats_1th = np.zeros((self.ivec_num, self.ivec_dim))
        self.stats_0th_row = np.zeros(self.ivec_num)
        self.stats_0th_col = np.zeros(self.ivec_num)

        self.loss = 0.0
        self.sv_num = 0

    def compute_accs(self, idx1_idx2_labels, pos_weight):
        x = self.ivecs
        for idx1, idx2, label in idx1_idx2_labels:
            score = 0.0
            score += 2 * np.dot(self.Lambda_x[idx1], x[idx2])
            score += self.xt_Gamma_x[idx1] + self.xt_Gamma_x[idx2]
            score += self.xt_c[idx1] + self.xt_c[idx2]
            score += self.k
            if score * label < 1:
                weight_label = -1.0  # for neg pair
                if 1 == label:
                    weight_label = pos_weight  # for pos pair
                # self.G[idx1, idx2] = label
                # self.stats_1th[idx1].add(label*x[idx2])
                np.add(self.stats_1th[idx1], weight_label *
                       x[idx2], out=self.stats_1th[idx1])
                self.stats_0th_row[idx1] += weight_label
                self.stats_0th_col[idx2] += weight_label
                self.loss += (1-score*label) * abs(weight_label)
                self.sv_num += 1

    def forward(self, idx1_idx2_labels, pos_weight):
        self.precompute()
        self.reset_accs()
        self.compute_accs(idx1_idx2_labels, pos_weight)

    def scoring(self, idx1_idx2s, idx2utt, opath2score):
        self.precompute()
        x = self.ivecs
        with open(opath2score, 'w') as fout:
            for idx1, idx2 in idx1_idx2s:
                score = 0.0
                score += 2 * np.dot(self.Lambda_x[idx1], x[idx2])
                score += self.xt_Gamma_x[idx1] + self.xt_Gamma_x[idx2]
                score += self.xt_c[idx1] + self.xt_c[idx2]
                score += self.k
                fout.write("%s %s %.6f\n" %
                           (idx2utt[idx1], idx2utt[idx2], score))
        print("trials are scored and saved to {0}".format(
            opath2score))
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def backward(self, lr, update_weight):
        cdef int ivec_num = self.ivec_num
        cdef int ivec_dim = self.ivec_dim
        cdef np.ndarray[DTYPE_t, ndim=2] x = self.ivecs
        cdef np.ndarray[DTYPE_t, ndim=2] delta_Lambda = np.dot(self.stats_1th.T, self.ivecs)
        cdef np.ndarray[DTYPE_t, ndim=1] sv_ys = self.stats_0th_col + self.stats_0th_row
        cdef np.ndarray[DTYPE_t, ndim=2] delta_Gamma = np.zeros((ivec_dim, ivec_dim))
        cdef np.ndarray[DTYPE_t, ndim=1] delta_c = np.zeros(ivec_dim)
        cdef np.ndarray[DTYPE_t, ndim=2] temp_Lambda, temp_Gamma
        cdef np.ndarray[DTYPE_t, ndim=1] temp_c
        cdef float delta_k, temp_k
        cdef int i, m, n

        delta_Lambda += delta_Lambda.T

        # sv_ys = self.stats_0th_col + self.stats_0th_row

        # delta_Gamma = np.zeros((self.ivec_dim, self.ivec_dim))
        # for i in range(ivec_num):
        #   delta_Gamma += sv_ys[i] * np.outer(x[i], x[i])

        for i in range(ivec_num):
            for m in range(ivec_dim):
                for n in range(m, ivec_dim):
                    delta_Gamma[m, n] += sv_ys[i] * x[i, m] * x[i, n]
        
        for m in range(ivec_dim):
            for n in range(m):
                delta_Gamma[m, n] = delta_Gamma[n, m]

        # delta_c = np.zeros(self.ivec_dim)
        # for i in range(self.ivec_num):
        #     delta_c += sv_ys[i] * x[i]

        for i in range(ivec_num):
            for m in range(ivec_dim):
                delta_c[m] += sv_ys[i] * x[i, m]

        delta_k = np.sum(self.stats_0th_col)

        # update paras
        temp_Lambda = self.Lambda * \
            (1 - lr) + lr * update_weight * delta_Lambda
        temp_Gamma = self.Gamma * (1 - lr) + lr * update_weight * delta_Gamma
        temp_c = self.c * (1 - lr) + lr * update_weight * delta_c
        temp_k = self.k * (1 - lr) + lr * update_weight * delta_k

        # will add gradient projection later
        self.Lambda = temp_Lambda
        self.Gamma = temp_Gamma
        self.c = temp_c
        self.k = temp_k

    def get_loss(self, penalty, batch_size):
        """ Just for diagnosizing
        """
        clf_loss = self.loss * penalty / batch_size
        para_loss = np.linalg.norm(self.Lambda)  # frobenius norm
        para_loss += np.linalg.norm(self.Gamma)
        para_loss += np.linalg.norm(self.c)
        para_loss += np.abs(self.k)
        print("clf loss is %.4f" % (clf_loss))
        print("para loss is %.4f" % (para_loss))
        print("sv_num: %d" % (self.sv_num))

    def save_model(self, opath2model, binary=True):
        from kaldi.util.io import xopen
        from kaldi.base.io import write_token, write_double
        from kaldi.matrix import DoubleVector, DoubleMatrix
        with xopen(opath2model, "wb") as ko:
            # pdb.set_trace()
            write_token(ko.stream(), binary, "<Lambda>")
            DoubleMatrix(self.Lambda).write(ko.stream(), binary)
            write_token(ko.stream(), binary, "<Gamma>")
            DoubleMatrix(self.Gamma).write(ko.stream(), binary)
            write_token(ko.stream(), binary, "<c>")
            DoubleVector(self.c).write(ko.stream(), binary)
            write_token(ko.stream(), binary, "<k>")
            write_double(ko.stream(), binary, self.k)

    def load_model(self, ipath2model):
        from kaldi.util.io import xopen
        from kaldi.base.io import expect_token, read_double
        from kaldi.matrix import DoubleVector, DoubleMatrix
        with xopen(ipath2model) as ki:
            expect_token(ki.stream(), ki.binary, "<Lambda>")
            Lambda = DoubleMatrix().read_(ki.stream(), ki.binary)
            expect_token(ki.stream(), ki.binary, "<Gamma>")
            Gamma = DoubleMatrix().read_(ki.stream(), ki.binary)
            expect_token(ki.stream(), ki.binary, "<c>")
            c = DoubleVector().read_(ki.stream(), ki.binary)
            expect_token(ki.stream(), ki.binary, "<k>")
            k = read_double(ki.stream(), ki.binary)

        self.Lambda = Lambda.numpy()
        self.Gamma = Gamma.numpy()
        self.c = c.numpy()
        self.k = k


if __name__ == '__main__':
    config = DataLoaderConfig()
    config.parse_args()  # revise configuration from cmd line
    config.print_args()
    from dataloader import PairwiseIvectorDataset
    from dataloader import Config
    from torch.utils.data import DataLoader
    batch_size = 1000000
    penalty = 32
    pos_weight = config.neg_pos_ratio
    opath2model = "exp/psvm/psvm1.mdl"

    if not os.path.exists(os.path.dirname(opath2model)):
        os.makedirs(os.path.dirname(opath2model))
    print('Start to load data...')
    start_time = time.time()
    dataset = PairwiseIvectorDataset(
        "ark:data/train/spk2utt",
        "scp:exp/ivectors_train/ivector.scp",
        config)
    et = time.time() - start_time
    et = str(datetime.timedelta(seconds=et))[:-7]
    print("Elapsed [{}]. Finish loading data.".format(et))

    # train_loader = DataLoader(dataset=dataset, batch_size=10,  shuffle=True, num_workers=1, drop_last=True)
    start_time = time.time()
    psvm = PSVM(dataset.ivecs)
    et = time.time() - start_time
    print('Elapsed [{}]. Finish model initialization'.format(et))
    # pdb.set_trace()
    # psvm.load_model(opath2model)

    random.shuffle(dataset.pairs) # slow

    # start training
    # forward
    lr_init = 0.3
    for i in range(int(len(dataset.pairs)/batch_size - 1)):
        lr = lr_init / (i+1)
        start_time = time.time()
        psvm.forward(dataset[batch_size*i:batch_size*(i+1)], pos_weight)

        et = time.time() - start_time
        print('Elapsed [{}]. Finish forward'.format(et))
        psvm.get_loss(penalty, batch_size)

        start_time = time.time()
        psvm.backward(lr, penalty*10/batch_size)
        et = time.time() - start_time
        print('Elapsed [{}]. Finish backward'.format(et))

        start_time = time.time()
        psvm.save_model(opath2model)
        et = time.time() - start_time
        print('Elapsed [{}]. Finish saving model'.format(et))
