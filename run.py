import argparse

from utils.config import load
from tools import cifar10_train, cifar10_train_daq, cifar10_train_qil, cifar10_train_qil_plus, cifar10_train_bit_prune
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def parse_args():
    parser = argparse.ArgumentParser(description='quantization network')
    parser.add_argument('--config', dest='config_file',
                        help='configuration filename',
                        default=None, type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = load(args.config_file)
    run_script = globals().get(config.solver)
    run_script.main(args)
