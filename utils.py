import pdb
import io
from collections import OrderedDict
from datetime import datetime
import subprocess
import threading
import kaldi_io
import sys


class Logger(object):
    def __init__(self, opath2logfile):
        self.terminal = sys.stdout
        self.log = open(opath2logfile, "w")
        self.write('Time: {0}\n'.format(str(datetime.now())))

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


class ConfigBase(object):
    def __new__(cls, *args, **kwargs):
        instance = object.__new__(cls)
        instance.__odict__ = OrderedDict()
        return instance

    def __setattr__(self, key, value):
        if key != '__odict__':
            self.__odict__[key] = value
        object.__setattr__(self, key, value)

    def print_args(self):
        """
        Print all configurations
        """
        print("[Configuration]")
        for key, value in self.__odict__.items():
            print('\'{0}\' : {1}'.format(key, value))
        print('')

    def parse_args(self):
        """
        Supports to pass arguments from command line
        """
        import argparse

        def str2bool(v):
            if v.lower() in ('true', 't'):
                return True
            elif v.lower() in ('false', 'f'):
                return False
            else:
                raise argparse.ArgumentTypeError('Boolean value expected.')
        parser = argparse.ArgumentParser()
        for key, value in self.__odict__.items():
            if bool == type(value):  # parser by default do not raise error for invalid bool args
                parser.add_argument('--'+key.replace("_", "-"),
                                    default=str(value), type=str2bool)
            else:
                parser.add_argument('--'+key.replace("_", "-"),
                                    default=value, type=type(value))
        args = parser.parse_args()
        args = vars(args)
        # update
        for key in self.__odict__:
            arg = args[key]
            self.__odict__[key] = arg
            object.__setattr__(self, key, arg)


def compute_eer(predict_result, trials, verbose=True):

    def cleanup(proc, cmd):
        ret = proc.wait()
        if ret > 0:
            raise SuprocessFailed('cmd %s returned %d !' % (cmd, ret))
        return

    # cmd = "compute-eer - 2>/dev/null"
    cmd = "compute-eer - "
    proc = subprocess.Popen(
        cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    threading.Thread(target=cleanup, args=(proc, cmd)).start()
    fout = io.TextIOWrapper(proc.stdin)
    fresult = io.TextIOWrapper(proc.stdout)
    try:
        with open(predict_result, 'r') as fscore, open(trials, 'r') as flabel:
            for idx, (line1, line2) in enumerate(zip(fscore, flabel)):
                pair1_1, pair1_2, score = line1.rstrip('\n').split(' ')
                pair2_1, pair2_2, label = line2.rstrip('\n').split(' ')
                assert pair1_1 == pair2_1, "Error while reading {predict_result} and {trials}. In line {line}, first field of {predict_result} doesn't match that of {trials}".format(
                    predict_result=predict_result,
                    trials=trials,
                    line=idx+1)
                assert pair1_2 == pair2_2, "Error while reading {predict_result} and {trials}. In line{line}, second filed of {predict_result} doesn't match that of {trials}".format(
                    predict_result=predict_result,
                    trials=trials,
                    line=idx+1)
                fout.write('{0} {1}\n'.format(score, label))
    finally:
        fout.close()

    eer = fresult.read()
    if verbose:
        print("EER is: {0}%".format(eer))
    return eer
