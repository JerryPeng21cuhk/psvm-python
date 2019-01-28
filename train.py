import pdb
import numpy as np
from dataloader import Config as DataLoaderConfig
import datetime
import time
import random
import os
from cmodel import PSVM


def main():
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

if __name__ == '__main__':
    main()
